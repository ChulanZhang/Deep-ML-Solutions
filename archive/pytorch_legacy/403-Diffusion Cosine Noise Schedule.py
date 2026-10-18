import torch

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> tuple:
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps)
    f_t = torch.cos(((t / timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_bar = f_t / f_t[0]
    
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = torch.clamp(betas, 0.0, 0.999)
    
    alphas = 1.0 - betas
    return torch.round(betas, 4), torch.round(alphas, 4), torch.round(alphas_bar[1:], 4)
