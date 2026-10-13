import torch

def score_matching_loss(score_pred: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor, std_t: float) -> float:
    score_pred = torch.tensor(score_pred, dtype=float)
    x_t = torch.tensor(x_t, dtype=float)
    x_0 = torch.tensor(x_0, dtype=float)
    
    target_score = -(x_t - x_0) / (std_t**2 + 1e-8)
    loss = torch.mean((score_pred - target_score)**2)
    return float(torch.round(loss, 4))
