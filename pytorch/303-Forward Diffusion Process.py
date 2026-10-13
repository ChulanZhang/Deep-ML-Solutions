import torch

def forward_diffusion(x0: torch.Tensor, t: int, alphas_cumprod: list, noise: torch.Tensor) -> torch.Tensor:
    x0 = torch.tensor(x0, dtype=float)
    noise = torch.tensor(noise, dtype=float)
    alpha_bar = alphas_cumprod[t]
    
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    return torch.round(xt, 4)
