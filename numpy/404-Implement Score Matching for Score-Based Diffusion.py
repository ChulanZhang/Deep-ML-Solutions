import numpy as np

def score_matching_loss(score_pred: np.ndarray, x_t: np.ndarray, x_0: np.ndarray, std_t: float) -> float:
    score_pred = np.array(score_pred, dtype=float)
    x_t = np.array(x_t, dtype=float)
    x_0 = np.array(x_0, dtype=float)
    
    target_score = -(x_t - x_0) / (std_t**2 + 1e-8)
    loss = np.mean((score_pred - target_score)**2)
    return float(np.round(loss, 4))
