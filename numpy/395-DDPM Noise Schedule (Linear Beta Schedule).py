import numpy as np

def linear_beta_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> tuple:
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return np.round(betas, 4), np.round(alphas, 4), np.round(alphas_cumprod, 4)
