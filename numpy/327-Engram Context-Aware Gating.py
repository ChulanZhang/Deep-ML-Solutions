import numpy as np

def engram_gating(x: np.ndarray, context: np.ndarray, W_x: np.ndarray, W_c: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    c = np.array(context, dtype=float)
    
    gate_logits = np.dot(x, W_x) + np.dot(c, W_c) + b
    gate = 1.0 / (1.0 + np.exp(-gate_logits))
    
    return np.round(gate * x, 4)
