import numpy as np

def gcn_layer(A: np.ndarray, H: np.ndarray, W: np.ndarray) -> np.ndarray:
    A = np.array(A, dtype=float)
    H = np.array(H, dtype=float)
    W = np.array(W, dtype=float)
    
    A_tilde = A + np.eye(A.shape[0])
    D_tilde = np.diag(np.sum(A_tilde, axis=1))
    
    D_inv_sqrt = np.linalg.inv(np.sqrt(D_tilde))
    
    out = D_inv_sqrt @ A_tilde @ D_inv_sqrt @ H @ W
    
    out = np.maximum(0, out)
    return np.round(out, 4)
