import numpy as np

def mdn_residualization(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=float)
    M = np.array(M, dtype=float)
    
    if M.ndim == 1:
        M = M[:, None]
        
    MtM_inv = np.linalg.pinv(M.T @ M)
    P = M @ MtM_inv @ M.T
    
    X_res = X - P @ X
    return np.round(X_res, 4)
