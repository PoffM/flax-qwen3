from transformers import AutoTokenizer
import jax.numpy as j
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from model import Qwen3Model
from convert_weights import convert_qwen3_params_for_linen
from safetensors import safe_open
import jax
import os

### Jax cache setup

jax_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".jax_cache")
jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


### Setup model

model_id = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_id)

snapshot_dir = snapshot_download(model_id)

print("Loading weights...")

cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
cfg["max_position_embeddings"] = 1024

src_weights = dict()
with safe_open(Path(snapshot_dir) / "model.safetensors", framework="flax") as f:
    for key in f.keys():
        src_weights[key] = f.get_tensor(key)

vars = {
  'params': convert_qwen3_params_for_linen(src_weights, cfg["num_hidden_layers"])
}

model = Qwen3Model(**cfg)

@jax.jit
def forward(vars: dict, tokens: jax.Array, pos=0, kv_cache: dict[int, jax.Array] = None):
  return model.apply(vars, tokens, pos, kv_cache)


### Chat Loop

input_len = cfg["max_position_embeddings"]
context = j.full((input_len,), tokenizer.pad_token_id)

pos = 0

while True:
  # Get user input
  prompt = input("Prompt: ")

  # Apply chat template (for user/assistant role tokens)
  prompt = tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
  )

  # Tokenize prompt
  prompt_tokens = j.array(tokenizer(prompt)['input_ids'])

  # Since the padding (or context) is prepended to meet the context length
  # in the 'prefill' stage, we roll back the position.
  # The padding tokens get negative positions, and the real input tokens start at 0.
  pos -= input_len

  # Sliding window model input
  context = j.concat([context, prompt_tokens])[-input_len:]
  input_tokens = context

  # Advance the position by the prompt token length
  pos += len(prompt_tokens)

  # Init kv cache
  kv_cache = j.zeros((cfg['num_hidden_layers'], 2, input_len, cfg['num_key_value_heads'], cfg['head_dim']), dtype=j.bfloat16)

  response_tokens = []

  # The Generation Loop:
  # The first iteration is the 'prefill' stage where we read the whole context,
  # and the subsequent iterations are the 'decode' stage where we only read the newest token.
  while True:
    # Forward pass
    logits, new_kv_cache = forward(vars, input_tokens, pos, kv_cache)
    # Update kv cache
    kv_cache = j.concat([kv_cache, new_kv_cache], axis=2)[:, :, -input_len:]

    # Choose the most likely token (highest logit)
    predicted_token = logits[-1].argmax().item()
    response_tokens.append(predicted_token)

    # Advance the position by the inputted token length
    pos += len(input_tokens)
    input_tokens = j.array([predicted_token])

    # Stop when the model outputs the end-of-sequence token
    if predicted_token == tokenizer.eos_token_id:
      break

    # Print the token text
    token_text = tokenizer.decode(predicted_token)
    print(token_text, end="", flush=True)

  # Update the context with the response tokens
  context = j.concat([context, j.array(response_tokens)])[-input_len:]

  print("")
