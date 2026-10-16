def context_parallelism_ring(seq_len: int, num_gpus: int, chunk_size: int) -> int:
    # In Ring Attention, seq is divided among GPUs.
    # Total chunks = seq_len / chunk_size
    # Chunks per GPU = Total / seq_len
    # Each GPU communicates its K,V chunks in a ring. 
    # Comm steps per GPU = num_gpus - 1
    
    return max(0, num_gpus - 1)
