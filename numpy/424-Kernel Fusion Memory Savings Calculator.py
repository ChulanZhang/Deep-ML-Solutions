def kernel_fusion_savings(batch_size: int, seq_len: int, hidden_size: int, num_intermediates: int) -> float:
    # Memory saved by not writing intermediates to HBM
    # size = batch_size * seq_len * hidden_size * 2 bytes (fp16)
    # Total saved = size * num_intermediates
    
    bytes_saved = batch_size * seq_len * hidden_size * 2 * num_intermediates
    mb_saved = bytes_saved / (1024 ** 2)
    return float(round(mb_saved, 4))
