import numpy as np

def per_channel_quantization(W: np.ndarray) -> tuple:
    W_arr = np.array(W, dtype=float)
    
    max_vals = np.max(np.abs(W_arr), axis=-1, keepdims=True)
    scales = 127.0 / (max_vals + 1e-12)
    
    quantized = np.clip(np.round(W_arr * scales), -128, 127).astype(np.int8)
    return quantized, np.round(scales, 4)
