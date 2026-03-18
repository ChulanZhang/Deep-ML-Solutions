import numpy as np

def ddim_step(xt: np.ndarray, pred_noise: np.ndarray, t: int, alphas_cumprod: list) -> np.ndarray:
    xt = np.array(xt, dtype=float)
    pred_noise = np.array(pred_noise, dtype=float)
    
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_t_prev = alphas_cumprod[t-1] if t > 0 else 1.0
    
    pred_x0 = (xt - np.sqrt(1 - alpha_bar_t) * pred_noise) / np.sqrt(alpha_bar_t)
    dir_xt = np.sqrt(1 - alpha_bar_t_prev) * pred_noise
    
    x_prev = np.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
    return np.round(x_prev, 4)
