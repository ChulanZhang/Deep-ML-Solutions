import torch

def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> float:
    a = torch.tensor(anchor, dtype=float)
    p = torch.tensor(positive, dtype=float)
    n = torch.tensor(negative, dtype=float)
    
    d_ap = np.linalg.norm(a - p, dim=-1)
    d_an = np.linalg.norm(a - n, dim=-1)
    
    loss = torch.maximum(0.0, d_ap - d_an + margin)
    return float(torch.round(torch.mean(loss), 4))
