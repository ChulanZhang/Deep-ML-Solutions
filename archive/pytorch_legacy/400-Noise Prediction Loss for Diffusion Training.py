import torch

def noise_prediction_loss(true_noise: torch.Tensor, pred_noise: torch.Tensor) -> float:
    true_noise = torch.tensor(true_noise, dtype=float)
    pred_noise = torch.tensor(pred_noise, dtype=float)
    mse = torch.mean((true_noise - pred_noise)**2)
    return float(torch.round(mse, 4))
