import torch

def mdn_residualization(f: torch.Tensor, X: torch.Tensor, sigma_inv: torch.Tensor, N: int) -> torch.Tensor:
    """
    Remove the linear influence of metadata X from features f.
    f_res = f - X(X^T X)^-1 X^T f
    """
    beta_hat = sigma_inv @ (X.T @ f)
    f_res = f - X @ beta_hat
    return f_res
