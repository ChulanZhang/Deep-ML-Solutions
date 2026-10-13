import torch

def ddim_step(xt: torch.Tensor, pred_noise: torch.Tensor, t: int, alphas_cumprod: list) -> torch.Tensor:
    xt = torch.tensor(xt, dtype=float)
    pred_noise = torch.tensor(pred_noise, dtype=float)
    
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_t_prev = alphas_cumprod[t-1] if t > 0 else 1.0
    
    pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * pred_noise
    
    x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
    return torch.round(x_prev, 4)
