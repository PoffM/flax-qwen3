import jax
import jax.numpy as j
import flax.linen as nn

class Qwen3Model(nn.Module):
  vocab_size: int
  hidden_size: int
  rms_norm_eps: float
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int
  head_dim: int
  rope_theta: float

  @nn.compact
  def __call__(self, x: jax.Array, pos=0):
    embeddings = nn.Embed(self.vocab_size, self.hidden_size)
    x = embeddings(x)

    T, C = x.shape

    for i in range(self.num_hidden_layers):
      lx = x

      # input norm
      lx_norm = nn.RMSNorm(self.rms_norm_eps, name=f"input_layernorm_{i}")(lx)
      
      # qkv projection
      q = nn.Dense(self.num_attention_heads * self.head_dim, use_bias=False, name=f"q_proj_{i}")(lx_norm)
      q = q.reshape(-1, self.num_attention_heads, self.head_dim)

      k = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name=f"k_proj_{i}")(lx_norm)
      k = k.reshape(-1, self.num_key_value_heads, self.head_dim)

      v = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name=f"v_proj_{i}")(lx_norm)
      v = v.reshape(-1, self.num_key_value_heads, self.head_dim)

      # QK norm
      q = nn.RMSNorm(self.rms_norm_eps, name=f"q_norm_{i}")(q)
      k = nn.RMSNorm(self.rms_norm_eps, name=f"k_norm_{i}")(k)
      
      # RoPE
      q = rope(q, self.rope_theta, pos)
      k = rope(k, self.rope_theta, pos)

      # attention
      attn_mask = j.tri(T, dtype=bool)
      attn_out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
      attn_out = attn_out.reshape(T, -1)

      # out projection
      o = nn.Dense(self.hidden_size, use_bias=False, name=f"o_proj_{i}")(attn_out)
      lx += o

      # post-attention norm
      lx_norm = nn.RMSNorm(self.rms_norm_eps, name=f"post_attention_layernorm_{i}")(lx)
      
      # MLP
      gate = jax.nn.silu(nn.Dense(3 * self.hidden_size, use_bias=False, name=f"gate_proj_{i}")(lx_norm))
      up = nn.Dense(3 * self.hidden_size, use_bias=False, name=f"up_proj_{i}")(lx_norm)
      lx += nn.Dense(self.hidden_size, use_bias=False, name=f"down_proj_{i}")(gate * up)

      x = lx

    # logits
    x = nn.RMSNorm(self.rms_norm_eps, name='norm')(x)
    logits = embeddings.attend(x)

    return logits

def rope(x, theta, pos=0):
    T, N, H = x.shape
    positions = pos + j.broadcast_to(j.arange(T), [T])
    freq = 1.0 / (theta ** (j.arange(0, H, 2, dtype=j.float32) / H))
    inp = j.einsum('t,h->th', positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = j.sin(inp).astype(x.dtype), j.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :H//2], x[:, :, H//2:]
    sin, cos = sin[:, None, :], cos[:, None, :] # [B, T, 1, H/2]
    return j.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
