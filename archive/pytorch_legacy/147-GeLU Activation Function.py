import torch

def GeLU(x: torch.Tensor) -> torch.Tensor:
    """Implement Gaussian Error Linear Unit"""
    x_arr = torch.tensor(x, dtype=float)
    # Using the tanh approximation
    inner = torch.sqrt(2.0 / np.pi) * (x_arr + 0.044715 * (x_arr ** 3))
    res = 0.5 * x_arr * (1.0 + np.tanh(inner))
    return torch.round(res, 4)
