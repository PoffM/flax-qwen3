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

# Jax cache
jax_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".jax_cache")
jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Model setup

model_id = "Qwen/Qwen3-0.6B-Base"
snapshot_dir = snapshot_download(model_id)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)

cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())

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
def forward(vars: dict, tokens: jax.Array):
  return model.apply(vars, tokens)

# Generate text in a loop

text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'The quick brown'

tokens = j.array(tokenizer(text)['input_ids'])

max_len = 50

print(text, end="", flush=True)
while len(tokens) < max_len:
  input = j.pad(tokens, (max_len - len(tokens), 0), constant_values=tokenizer.pad_token_id)
  logits = forward(vars, input)

  next_token = logits[-1].argmax().item()
  next_text = tokenizer.decode(next_token)

  text += next_text
  tokens = j.concat([tokens, j.array([next_token])])
  print(next_text, end="", flush=True)

print("")