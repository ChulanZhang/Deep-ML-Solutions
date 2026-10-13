import torch

def activation_derivatives(x: torch.Tensor, act_type: str) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    if act_type == 'relu':
        return (x > 0).to(torch.float32)
    elif act_type == 'sigmoid':
        s = 1.0 / (1.0 + torch.exp(-x))
        return torch.round(s * (1.0 - s), 4)
    elif act_type == 'tanh':
        t = np.tanh(x)
        return torch.round(1.0 - t**2, 4)
    return torch.zeros_like(x)
