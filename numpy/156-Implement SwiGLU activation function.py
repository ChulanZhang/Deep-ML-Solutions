import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    x_arr = np.array(x, dtype=float)
    d = x_arr.shape[-1] // 2
    x1, x2 = x_arr[..., :d], x_arr[..., d:]
    
    # Swish = x1 * sigmoid(x1)
    swish = x1 / (1.0 + np.exp(-x1))
    
    return np.round(swish * x2, 4)
