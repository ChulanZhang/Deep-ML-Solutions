import numpy as np

def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply RMSNorm to the input array.
    """
    ms = np.mean(x**2, axis=-1, keepdims=True)
    norm_x = x / np.sqrt(ms + eps)
    return g * norm_x
