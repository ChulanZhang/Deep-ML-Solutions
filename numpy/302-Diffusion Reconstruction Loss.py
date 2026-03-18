import numpy as np

def diffusion_reconstruction_loss(true_noise: np.ndarray, pred_noise: np.ndarray) -> float:
    true_noise = np.array(true_noise, dtype=float)
    pred_noise = np.array(pred_noise, dtype=float)
    mse = np.mean((true_noise - pred_noise)**2)
    return float(np.round(mse, 4))
