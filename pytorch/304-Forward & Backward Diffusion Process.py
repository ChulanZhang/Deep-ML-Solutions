import torch

def diffusion_processes(x0, t, alphas, alphas_cumprod, noise, pred_noise, z):
    x0 = torch.tensor(x0, dtype=float)
    noise = torch.tensor(noise, dtype=float)
    pred_noise = torch.tensor(pred_noise, dtype=float)
    z = torch.tensor(z, dtype=float)
    
    # Forward step
    alpha_bar = alphas_cumprod[t]
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    
    # Backward step
    alpha = alphas[t]
    
    if t > 0:
        sigma_t = torch.sqrt((1 - alphas_cumprod[t-1]) / (1 - alpha_bar) * (1 - alpha))
    else:
        sigma_t = 0.0
        
    x_prev = (1.0 / torch.sqrt(alpha)) * (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise)
    if t > 0:
        x_prev += sigma_t * z
        
    return torch.round(xt, 4), torch.round(x_prev, 4)
