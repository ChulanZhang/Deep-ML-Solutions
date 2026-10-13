import torch

def hardtanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    return torch.clamp(x, -1.0, 1.0)
