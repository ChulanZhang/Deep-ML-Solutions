import torch

def classifier_free_guidance(uncond_noise: torch.Tensor, cond_noise: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    uncond = torch.tensor(uncond_noise, dtype=float)
    cond = torch.tensor(cond_noise, dtype=float)
    
    noise = uncond + guidance_scale * (cond - uncond)
    return torch.round(noise, 4)
