import numpy as np

def qk_norm(Q: np.ndarray, K: np.ndarray) -> tuple:
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    
    Q_norm = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-12)
    K_norm = K / (np.linalg.norm(K, axis=-1, keepdims=True) + 1e-12)
    
    return np.round(Q_norm, 4), np.round(K_norm, 4)
