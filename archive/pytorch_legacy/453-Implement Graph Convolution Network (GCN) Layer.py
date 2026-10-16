import torch

def gcn_layer(A: torch.Tensor, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    A = torch.tensor(A, dtype=float)
    H = torch.tensor(H, dtype=float)
    W = torch.tensor(W, dtype=float)
    
    A_tilde = A + torch.eye(A.shape[0])
    D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
    
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(D_tilde))
    
    out = D_inv_sqrt @ A_tilde @ D_inv_sqrt @ H @ W
    
    out = torch.maximum(0, out)
    return torch.round(out, 4)
