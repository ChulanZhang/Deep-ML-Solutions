import torch

def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = torch.tensor(y_true, dtype=float)
    y_pred = torch.tensor(y_pred, dtype=float)
    y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
    loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return float(torch.round(loss, 4))
