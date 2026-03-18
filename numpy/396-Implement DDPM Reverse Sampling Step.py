import numpy as np

def ddpm_reverse_step(xt: np.ndarray, pred_noise: np.ndarray, t: int, alphas: list, alphas_cumprod: list, betas: list, z: np.ndarray) -> np.ndarray:
    xt = np.array(xt, dtype=float)
    pred_noise = np.array(pred_noise, dtype=float)
    z = np.array(z, dtype=float)
    
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    beta_t = betas[t]
    
    x_prev = (1.0 / np.sqrt(alpha_t)) * (xt - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * pred_noise)
    
    if t > 0:
        sigma_t = np.sqrt(beta_t)
        x_prev += sigma_t * z
        
    return np.round(x_prev, 4)
