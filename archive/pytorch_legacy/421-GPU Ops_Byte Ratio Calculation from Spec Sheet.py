def gpu_ops_byte_ratio(mac_tops: float, mem_bw_gbs: float) -> float:
    # mac_tops: MAC operations in Tera (1e12 MACs = 2e12 FLOPs)
    # mem_bw_gbs: Memory bandwidth in GiB/s (1024**3 bytes/s) or GB/s (1e9 bytes/s).
    # Deep-ML implies FLOPs/Byte = (mac_tops * 2 * 1e12) / (mem_bw_gbs * 1e9)
    flops = mac_tops * 2 * 1e12
    bytes_bw = mem_bw_gbs * 1e9
    
    ratio = flops / bytes_bw
    return float(round(ratio, 4))
