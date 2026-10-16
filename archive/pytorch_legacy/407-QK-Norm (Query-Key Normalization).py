import torch

def qk_norm(Q: torch.Tensor, K: torch.Tensor) -> tuple:
    Q = torch.tensor(Q, dtype=float)
    K = torch.tensor(K, dtype=float)
    
    Q_norm = Q / (np.linalg.norm(Q, dim=-1, keepdim=True) + 1e-12)
    K_norm = K / (np.linalg.norm(K, dim=-1, keepdim=True) + 1e-12)
    
    return torch.round(Q_norm, 4), torch.round(K_norm, 4)
