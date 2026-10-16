import torch

def mdn_residualization(X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    X = torch.tensor(X, dtype=float)
    M = torch.tensor(M, dtype=float)
    
    if M.ndim == 1:
        M = M[:, None]
        
    MtM_inv = np.linalg.pinv(M.T @ M)
    P = M @ MtM_inv @ M.T
    
    X_res = X - P @ X
    return torch.round(X_res, 4)
