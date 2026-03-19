def tensor_parallel_allreduce(tensor_size_mb: float, bw_gbps: float, num_gpus: int) -> float:
    # All-Reduce time using optimal ring/tree for TP
    # Data Volume in All-Reduce = 2 * (N-1) / N * data
    data_volume = 2 * (num_gpus - 1) / num_gpus * tensor_size_mb
    
    bw_mbps = bw_gbps * 1024
    latency_s = data_volume / bw_mbps
    return float(round(latency_s * 1000, 4)) # ms
