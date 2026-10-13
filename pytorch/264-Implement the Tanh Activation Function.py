import torch

def tanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    res = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    return torch.round(res, 4)
