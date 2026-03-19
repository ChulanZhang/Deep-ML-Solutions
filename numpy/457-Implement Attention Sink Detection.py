import numpy as np

def detect_attention_sinks(attention_weights: np.ndarray, sink_size: int = 4, threshold: float = 0.5) -> bool:
    attn = np.array(attention_weights, dtype=float)
    avg_attn_to_keys = np.mean(attn, axis=(0, 1, 2))
    sink_attn = np.sum(avg_attn_to_keys[:sink_size])
    return bool(sink_attn >= threshold)
