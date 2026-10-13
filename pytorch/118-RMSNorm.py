import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply RMSNorm to the input array.
    """
    ms = torch.mean(x**2, dim=-1, keepdim=True)
    norm_x = x / torch.sqrt(ms + eps)
    return g * norm_x
