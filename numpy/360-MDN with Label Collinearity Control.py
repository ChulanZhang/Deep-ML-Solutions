import numpy as np

def mdn_collinearity_control(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=float)
    M = np.array(M, dtype=float)
    Y = np.array(Y, dtype=float)
    
    if Y.ndim == 1:
        Y = Y[:, None]
    if M.ndim == 1:
        M = M[:, None]
        
    YtY_inv = np.linalg.pinv(Y.T @ Y)
    P_Y = Y @ YtY_inv @ Y.T
    M_res = M - P_Y @ M
    
    MtM_inv = np.linalg.pinv(M_res.T @ M_res)
    P_M = M_res @ MtM_inv @ M_res.T
    
    X_res = X - P_M @ X
    return np.round(X_res, 4)
