import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float = 0.1, seed: int = 42) -> list[float]:
    np.random.seed(seed)
    x_arr = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)
    
    # Linear 1
    hidden = np.dot(x_arr, W1) + b1
    # ReLU
    hidden = np.maximum(0, hidden)
    # Dropout
    if dropout_p > 0:
        mask = (np.random.rand(*hidden.shape) >= dropout_p).astype(float)
        hidden = hidden * mask / (1.0 - dropout_p)
        
    # Linear 2
    out = np.dot(hidden, W2) + b2
    
    # Residual
    res = out + x_arr
    return res.tolist()
