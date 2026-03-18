import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
    """Implement Gaussian Error Linear Unit"""
    x_arr = np.array(x, dtype=float)
    # Using the tanh approximation
    inner = np.sqrt(2.0 / np.pi) * (x_arr + 0.044715 * (x_arr ** 3))
    res = 0.5 * x_arr * (1.0 + np.tanh(inner))
    return np.round(res, 4)
