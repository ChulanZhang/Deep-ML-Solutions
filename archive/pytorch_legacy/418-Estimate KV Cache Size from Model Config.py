def estimate_kv_cache_size(batch_size: int, seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, precision: str = 'fp16') -> float:
    byte_map = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1}
    b_per_elem = byte_map.get(precision.lower(), 2)
    
    # Every token requires 1 Key and 1 Value vector per layer
    elements_per_token = 2 * num_layers * num_kv_heads * head_dim
    bytes_per_token = elements_per_token * b_per_elem
    
    total_bytes = batch_size * seq_len * bytes_per_token
    
    # Return size in MB
    return float(round(total_bytes / (1024 ** 2), 4))
