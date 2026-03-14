import os
import jax
import jax.numpy as j
from huggingface_hub import snapshot_download, scan_cache_dir
from transformers import PreTrainedTokenizerFast
from safetensors import safe_open
import json
from pathlib import Path
from model import Qwen3Model
from convert_weights import convert_qwen3_params_for_linen
import sys

### Jax cache setup

jax_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".jax_cache")
jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

### Model setup

model_id = "Qwen/Qwen3-0.6B-Base"
snapshot_dir = snapshot_download(model_id)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)

cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
cfg["max_position_embeddings"] = 50

print("Loading weights...")

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

ctx_len = cfg["max_position_embeddings"]

text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'The quick brown'

# Init kv cache
kv_cache: dict[int, jax.Array] = dict()
for i in range(cfg["num_hidden_layers"]):
  kv_cache[i] = j.zeros((2, ctx_len, cfg['num_key_value_heads'], cfg['head_dim']), dtype=j.float32)

print(text, end="", flush=True)

### Do an initial 'prefill' inference to populate the kv cache with the initial tokens' keys and values.

tokens = j.array(tokenizer(text)['input_ids'])
input = j.pad(tokens, (ctx_len - len(tokens), 0), constant_values=tokenizer.pad_token_id)
# The padding tokens get negative positions, and the real input tokens start at 0.
pos = len(tokens) - ctx_len
logits, kv_cache = forward(vars, input, pos, kv_cache)
pos = len(tokens)

predicted_token = logits[-1].argmax().item()
next_text = tokenizer.decode(predicted_token)
text += next_text
print(next_text, end="", flush=True)

### Predict the next token in a loop, only passing the last predicted token.

while pos < ctx_len:
  input = j.array([predicted_token])
  logits, kv_cache = forward(vars, input, pos, kv_cache)

  predicted_token = logits[-1].argmax().item()
  next_text = tokenizer.decode(predicted_token)
  text += next_text
  print(next_text, end="", flush=True)

  pos += 1

print("")