import numpy as np

def activation_derivatives(x: np.ndarray, act_type: str) -> np.ndarray:
    x = np.array(x, dtype=float)
    if act_type == 'relu':
        return (x > 0).astype(float)
    elif act_type == 'sigmoid':
        s = 1.0 / (1.0 + np.exp(-x))
        return np.round(s * (1.0 - s), 4)
    elif act_type == 'tanh':
        t = np.tanh(x)
        return np.round(1.0 - t**2, 4)
    return np.zeros_like(x)
