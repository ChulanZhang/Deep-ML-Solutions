import numpy as np

def mdn_with_collinearity(f: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          sigma_tilde_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Remove metadata effects while preserving label-relevant information.
    """
    y_col = y if y.ndim == 2 else y.reshape(-1, 1)
    X_tilde = np.concatenate([X, y_col], axis=1)
    
    beta_tilde = sigma_tilde_inv @ (X_tilde.T @ f)
    
    K = X.shape[1]
    beta_X = beta_tilde[:K, :]
    
    f_res = f - X @ beta_X
    return f_res
