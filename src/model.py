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
  def __call__(self, x: jax.Array, pos=0, kv_cache: dict[int, jax.Array] = None):
    embeddings = nn.Embed(self.vocab_size, self.hidden_size)
    x = embeddings(x)

    T, C = x.shape

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

      if self.use_cache:
        kv_cache = { **kv_cache, i: j.concat([kv_cache[i][:, T:], j.stack([k, v])], axis=1) }
        # kv_cache = { **kv_cache, i: jax.lax.dynamic_update_slice(kv_cache[i], j.stack([k, v]), (0, pos, 0, 0)) }
        k, v = kv_cache[i]

      # grouped query attention
      attn_mask = j.tri(self.max_position_embeddings, dtype=bool)[-T:]
      attn_out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
      attn_out = attn_out.reshape(T, -1)

      # out projection
      o = nn.Dense(self.hidden_size, use_bias=False, name=f"o_proj_{i}")(attn_out)
      x += o

      # post-attention norm
      x_norm = nn.RMSNorm(self.rms_norm_eps, name=f"post_attention_layernorm_{i}")(x)
      
      # MLP
      gate = nn.Dense(self.intermediate_size, use_bias=False, name=f"gate_proj_{i}")(x_norm)
      gate = jax.nn.silu(gate)
      up = nn.Dense(self.intermediate_size, use_bias=False, name=f"up_proj_{i}")(x_norm)
      down = nn.Dense(self.hidden_size, use_bias=False, name=f"down_proj_{i}")(gate * up)

      # add to residual
      x += down

    # logits
    x = nn.RMSNorm(self.rms_norm_eps, name='norm')(x)
    logits = embeddings.attend(x)

    return logits, kv_cache

def rope(x, theta, pos=0):
    T, N, H = x.shape
    positions = pos + j.arange(T) # (T)
    freq = 1.0 / (theta ** (j.arange(0, H, 2) / H)) # (H/2)
    angles = positions[:, None] * freq # (T, H/2)
    sin = j.sin(angles)[:, None, :] # (T, 1, H/2)
    cos = j.cos(angles)[:, None, :] # (T, 1, H/2)
    x1, x2 = j.split(x, 2, axis=-1) # 2x (T, N, H/2)
    rotated = j.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1) # (T, N, H)
    return rotated
