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
'/home/mat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd'

# Model setup

model_id = "Qwen/Qwen3-0.6B-Base"
snapshot_dir = snapshot_download(model_id)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)

cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())

src_weights = dict()
with safe_open(Path(snapshot_dir) / "model.safetensors", framework="flax") as f:
    for key in f.keys():
        src_weights[key] = f.get_tensor(key)

vars = {
  'params': convert_qwen3_params_for_linen(src_weights, cfg["num_hidden_layers"])
}

model = Qwen3Model(
  vocab_size=cfg["vocab_size"],
  hidden_size=cfg["hidden_size"],
  rms_norm_eps=cfg["rms_norm_eps"],
  num_hidden_layers=cfg["num_hidden_layers"],
  num_attention_heads=cfg['num_attention_heads'],
  num_key_value_heads=cfg['num_key_value_heads'],
  head_dim=cfg['head_dim'],
  rope_theta=cfg['rope_theta'],
)

# @jax.jit
def forward(tokens: jax.Array):
  return model.apply(vars, tokens)

# Generate text in a loop

text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'The quick brown'

tokens = j.array(tokenizer(text)['input_ids'])

print(text)
for i in range(10):
  logits = forward(tokens)

  next_token = logits[-1].argmax().item()
  next_text = tokenizer.decode(next_token)

  text += next_text
  tokens = j.concat([tokens, j.array([next_token])])
  print(next_text)
