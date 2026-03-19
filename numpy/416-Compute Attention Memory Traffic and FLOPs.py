import numpy as np

def compute_attention_memory_traffic(batch_size: int, seq_len: int, num_heads: int, head_dim: int, bytes_per_elem: int = 2) -> tuple:
    # Traffic per attention mechanism:
    # Read Q, K, V -> 3 * (B * N * S * D)
    # Write Out -> (B * N * S * D)
    # Total -> 4 Tensor reads/writes of identical shapes assuming standard execution map
    # Note: KV-Cache implementations might vary but standard self-attention reads them completely inside MHA
    
    elements_per_tensor = batch_size * num_heads * seq_len * head_dim
    bytes_per_tensor = elements_per_tensor * bytes_per_elem
    total_memory_traffic = 4 * bytes_per_tensor
    
    # Approximate MACs = 2 operations per MAC, specifically for QK^T and Attention*V
    # QK^T: B * N * S * S * D MACs
    # AttnV: B * N * S * S * D MACs
    mac_ops = 2 * (batch_size * num_heads * (seq_len ** 2) * head_dim)
    flops = 2 * mac_ops # 2 flops per MAC
    
    return int(total_memory_traffic), int(flops)
