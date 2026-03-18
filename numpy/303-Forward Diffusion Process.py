import numpy as np

def forward_diffusion(x0: np.ndarray, t: int, alphas_cumprod: list, noise: np.ndarray) -> np.ndarray:
    x0 = np.array(x0, dtype=float)
    noise = np.array(noise, dtype=float)
    alpha_bar = alphas_cumprod[t]
    
    xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * noise
    return np.round(xt, 4)
