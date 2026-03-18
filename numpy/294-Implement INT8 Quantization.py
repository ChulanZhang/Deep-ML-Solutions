import numpy as np

def int8_quantize(W: np.ndarray) -> tuple:
    W_arr = np.array(W, dtype=float)
    max_val = np.max(np.abs(W_arr))
    if max_val == 0:
        return W_arr.astype(np.int8), 1.0
    scale = 127.0 / max_val
    W_quant = np.clip(np.round(W_arr * scale), -128, 127).astype(np.int8)
    return W_quant, float(np.round(scale, 4))
