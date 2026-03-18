import numpy as np

def mdn_residualization(f: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Remove the linear influence of metadata X from features f.
    f_res = f - X(X^T X)^-1 X^T f
    """
    beta_hat = sigma_inv @ (X.T @ f)
    f_res = f - X @ beta_hat
    return f_res
