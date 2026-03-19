def multi_gpu_communication_overhead(tensor_size_mb: float, bw_gbps: float, num_gpus: int, latency_us: float) -> float:
    # Communicating tensor_size_mb across rings/trees.
    # Usually standard all-reduce sends (2*(N-1)/N) * data
    # We follow the standard formula here: T = Latency + (Data_Volume / Bandwidth)
    
    # Volume transferred by a node in ring all-reduce:
    volume_mb = 2 * (num_gpus - 1) * tensor_size_mb / num_gpus
    
    # Bandwidth in MB/s
    bw_mbps = bw_gbps * 1024
    
    # Latency in seconds
    lat_s = latency_us * 1e-6
    
    time_s = lat_s + (volume_mb / bw_mbps)
    return float(round(time_s * 1000, 4))  # Return milliseconds
