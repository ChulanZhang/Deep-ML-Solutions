"""Deep-ML #418: Estimate KV Cache Size from Model Config.
https://www.deep-ml.com/problems/418
Standalone PyTorch solution; public-example replay is not an official submission.
This exercise uses binary KiB/MiB divisors under the labels KB/MB; excludes allocator overhead.
"""
def estimate_kv_cache_size(model_config, batch_size, seq_len):
    layers = model_config['num_layers']
    heads = model_config['num_attention_heads']
    kv_heads = model_config.get('num_kv_heads', heads)
    hidden = model_config['hidden_size']
    size = model_config['dtype_bytes']
    if min(layers, heads, kv_heads, hidden, size) <= 0 or hidden % heads or heads % kv_heads:
        raise ValueError('Invalid model dimensions.')
    if batch_size < 0 or seq_len < 0:
        raise ValueError('Batch and sequence lengths must be non-negative.')
    per_token = 2 * layers * kv_heads * (hidden // heads)
    elements = per_token * batch_size * seq_len
    nbytes = elements * size
    return {'kv_cache_elements': elements, 'kv_cache_size_bytes': nbytes,
            'kv_cache_size_mb': round(nbytes / 1024**2, 4),
            'per_layer_size_mb': round(nbytes / layers / 1024**2, 4),
            'per_token_size_kb': round(per_token * size / 1024, 4)}
