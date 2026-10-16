import torch

def vae_loss(recon_loss: float, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0) -> float:
    mu = torch.tensor(mu, dtype=float)
    log_var = torch.tensor(log_var, dtype=float)
    
    kl_div = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=-1)
    mean_kl = torch.mean(kl_div)
    
    total_loss = float(recon_loss) + beta * mean_kl
    return float(torch.round(total_loss, 4))
