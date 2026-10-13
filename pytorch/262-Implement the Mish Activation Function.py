import torch

def mish(x: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    softplus = np.log1p(torch.exp(x))
    res = x * np.tanh(softplus)
    return torch.round(res, 4)
