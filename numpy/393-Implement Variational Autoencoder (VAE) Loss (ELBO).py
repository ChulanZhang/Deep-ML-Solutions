import numpy as np

def vae_loss(recon_loss: float, mu: np.ndarray, log_var: np.ndarray, beta: float = 1.0) -> float:
    mu = np.array(mu, dtype=float)
    log_var = np.array(log_var, dtype=float)
    
    kl_div = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=-1)
    mean_kl = np.mean(kl_div)
    
    total_loss = float(recon_loss) + beta * mean_kl
    return float(np.round(total_loss, 4))
