def context_compression_ratio(original_seq_len: int, compressed_seq_len: int) -> float:
    return float(round(original_seq_len / compressed_seq_len, 4))
