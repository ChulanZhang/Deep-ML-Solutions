import numpy as np

def tanh(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    res = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    return np.round(res, 4)
