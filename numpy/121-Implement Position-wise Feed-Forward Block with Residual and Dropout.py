import numpy as np

def ffn(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, dropout_p: float, seed: int) -> np.ndarray:
    """
    Position-wise Feed-Forward Network (FFN) with residual connection and dropout.
    """
    np.random.seed(seed)
    
    # 1. Linear 1 + ReLU
    inter = np.maximum(0, np.dot(x, w1) + b1)
    
    # 2. Dropout
    mask = (np.random.rand(*inter.shape) >= dropout_p).astype(float)
    dropped = inter * mask / (1.0 - dropout_p)
    
    # 3. Linear 2
    out = np.dot(dropped, w2) + b2
    
    # 4. Residual and Round
    result = np.round(out + x, 4)
    
    return result
