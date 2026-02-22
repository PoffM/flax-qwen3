def convert_qwen3_params_for_linen(src_weights: dict, num_hidden_layers: int):
  params = {
    'Embed_0': {
      'embedding': src_weights['model.embed_tokens.weight']
    },
    'norm': {
      'scale': src_weights['model.norm.weight'],
    }
  }

  for i in range(num_hidden_layers):
    params[f'input_layernorm_{i}'] = {
      'scale': src_weights[f'model.layers.{i}.input_layernorm.weight']
    }
    params[f'q_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.self_attn.q_proj.weight'].transpose(1, 0),
    }
    params[f'k_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.self_attn.k_proj.weight'].transpose(1, 0),
    }
    params[f'v_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.self_attn.v_proj.weight'].transpose(1, 0),
    }
    params[f'q_norm_{i}'] = {
      'scale': src_weights[f'model.layers.{i}.self_attn.q_norm.weight'],
    }
    params[f'k_norm_{i}'] = {
      'scale': src_weights[f'model.layers.{i}.self_attn.k_norm.weight'],
    }
    params[f'o_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.self_attn.o_proj.weight'].transpose(1, 0),
    }
    params[f'post_attention_layernorm_{i}'] = {
      'scale': src_weights[f'model.layers.{i}.post_attention_layernorm.weight'],
    }
    params[f'gate_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.mlp.gate_proj.weight'].transpose(1, 0),
    }
    params[f'up_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.mlp.up_proj.weight'].transpose(1, 0),
    }
    params[f'down_proj_{i}'] = {
      'kernel': src_weights[f'model.layers.{i}.mlp.down_proj.weight'].transpose(1, 0),
    }

  return params

