import torch

def ddpm_reverse_step(xt: torch.Tensor, pred_noise: torch.Tensor, t: int, alphas: list, alphas_cumprod: list, betas: list, z: torch.Tensor) -> torch.Tensor:
    xt = torch.tensor(xt, dtype=float)
    pred_noise = torch.tensor(pred_noise, dtype=float)
    z = torch.tensor(z, dtype=float)
    
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    beta_t = betas[t]
    
    x_prev = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
    
    if t > 0:
        sigma_t = torch.sqrt(beta_t)
        x_prev += sigma_t * z
        
    return torch.round(x_prev, 4)
