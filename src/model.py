import jax
import jax.numpy as j
import flax.linen as nn

class Qwen3Model(nn.Module):
  architectures: list[str]
  attention_bias: bool
  attention_dropout: float
  bos_token_id: int
  eos_token_id: int
  head_dim: int
  hidden_act: str
  hidden_size: int
  initializer_range: float
  intermediate_size: int
  max_position_embeddings: int
  max_window_layers: int
  model_type: str
  num_attention_heads: int
  num_hidden_layers: int
  num_key_value_heads: int
  rms_norm_eps: float
  rope_scaling: any
  rope_theta: int
  sliding_window: any
  tie_word_embeddings: bool
  torch_dtype: str
  transformers_version: str
  use_cache: bool
  use_sliding_window: bool
  vocab_size: int

  @nn.compact
  def __call__(self, x: jax.Array, pos=0, kv_cache: jax.Array = None):
    embeddings = nn.Embed(self.vocab_size, self.hidden_size)
    x = embeddings(x)

    T, C = x.shape

    # Mask out the leading padding tokens,
    # so actual word tokens only attend to each other.
    # Negative positions are for padding tokens.
    # Example when pos=-2 and we have 2 padding + 3 word tokens:
    #   0 0 0 0 0
    #   0 0 0 0 0
    #   0 0 1 1 1
    #   0 0 0 1 1
    #   0 0 0 0 1
    ctx_start_pos = pos + T - self.max_position_embeddings
    ctx_positions = j.arange(self.max_position_embeddings) + ctx_start_pos
    attn_mask = j.where(ctx_positions >= 0, j.tri(self.max_position_embeddings, dtype=j.bool), False)
    attn_mask = attn_mask[-T:]

    new_kv_cache = j.zeros((self.num_hidden_layers, 2, T, self.num_key_value_heads, self.head_dim), dtype=j.bfloat16)

    for i in range(self.num_hidden_layers):
      # input norm
      x_norm = nn.RMSNorm(self.rms_norm_eps, name=f"input_layernorm_{i}")(x)

      # qkv projection
      q = nn.Dense(self.num_attention_heads * self.head_dim, use_bias=False, name=f"q_proj_{i}")(x_norm)
      q = q.reshape(-1, self.num_attention_heads, self.head_dim)

      k = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name=f"k_proj_{i}")(x_norm)
      k = k.reshape(-1, self.num_key_value_heads, self.head_dim)

      v = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name=f"v_proj_{i}")(x_norm)
      v = v.reshape(-1, self.num_key_value_heads, self.head_dim)

      # QK norm
      q = nn.RMSNorm(self.rms_norm_eps, name=f"q_norm_{i}")(q)
      k = nn.RMSNorm(self.rms_norm_eps, name=f"k_norm_{i}")(k)
      
      # RoPE
      q = rope(q, self.rope_theta, pos)
      k = rope(k, self.rope_theta, pos)

      # KV cache
      if self.use_cache:
        cache_read = kv_cache[i, :, T:]
        new_kv_cache = new_kv_cache.at[i].set(j.stack([k, v]))
        k, v = j.concat([cache_read, new_kv_cache[i]], axis=1)

      # grouped query attention
      attn_out = jax.nn.dot_product_attention(
        # Convert args to float32 to avoid this error when running on my 1070:
        # UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32
        q.astype(j.float32),
        k.astype(j.float32),
        v.astype(j.float32),
        mask=attn_mask
      ).astype(j.bfloat16)
      attn_out = attn_out.reshape(T, -1)

      # out projection
      attn_out_proj = nn.Dense(self.hidden_size, use_bias=False, name=f"o_proj_{i}")(attn_out)
      x += attn_out_proj

      # post-attention norm
      x_norm = nn.RMSNorm(self.rms_norm_eps, name=f"post_attention_layernorm_{i}")(x)
      
      # Gated MLP
      gate = nn.Dense(self.intermediate_size, use_bias=False, name=f"gate_proj_{i}")(x_norm)
      gate = jax.nn.silu(gate)
      value = nn.Dense(self.intermediate_size, use_bias=False, name=f"up_proj_{i}")(x_norm)
      gated_value = gate * value
      gate_out = nn.Dense(self.hidden_size, use_bias=False, name=f"down_proj_{i}")(gated_value)

      # add to residual
      x += gate_out

    # logits
    x = nn.RMSNorm(self.rms_norm_eps, name='norm')(x)
    logits = embeddings.attend(x)

    return logits, new_kv_cache

def rope(x: jax.Array, theta, pos=0):
    T, N, H = x.shape
    positions = pos + j.arange(T, dtype=j.bfloat16) # (T)
    freq = 1.0 / (theta ** (j.arange(0, H, 2, dtype=j.bfloat16) / H)) # (H/2)
    angles = positions[:, None] * freq # (T, H/2)
    sin = j.sin(angles)[:, None, :] # (T, 1, H/2)
    cos = j.cos(angles)[:, None, :] # (T, 1, H/2)
    x1, x2 = j.split(x, 2, axis=-1) # 2x (T, N, H/2)
    rotated = j.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1) # (T, N, H)
    return rotated
