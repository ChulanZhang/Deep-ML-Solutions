import torch

def latent_diffusion_codec(x: torch.Tensor, E: torch.Tensor, D: torch.Tensor) -> tuple:
    x = torch.tensor(x, dtype=float)
    E = torch.tensor(E, dtype=float)
    D = torch.tensor(D, dtype=float)
    
    if x.ndim == 2:
        latent = torch.matmul(x, E)
        reconstructed = torch.matmul(latent, D)
    else:
        # standard 1D fallback
        latent = torch.matmul(x, E)
        reconstructed = torch.matmul(latent, D)
        
    return torch.round(latent, 4), torch.round(reconstructed, 4)
