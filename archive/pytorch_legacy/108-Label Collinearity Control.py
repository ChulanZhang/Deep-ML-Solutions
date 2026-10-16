import torch

def mdn_with_collinearity(f: torch.Tensor, X: torch.Tensor, y: torch.Tensor, 
                          sigma_tilde_inv: torch.Tensor, N: int) -> torch.Tensor:
    """
    Remove metadata effects while preserving label-relevant information.
    """
    y_col = y if y.ndim == 2 else y.reshape(-1, 1)
    X_tilde = torch.cat([X, y_col], dim=1)
    
    beta_tilde = sigma_tilde_inv @ (X_tilde.T @ f)
    
    K = X.shape[1]
    beta_X = beta_tilde[:K, :]
    
    f_res = f - X @ beta_X
    return f_res
