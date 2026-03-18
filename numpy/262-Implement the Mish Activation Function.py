import numpy as np

def mish(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    softplus = np.log1p(np.exp(x))
    res = x * np.tanh(softplus)
    return np.round(res, 4)
