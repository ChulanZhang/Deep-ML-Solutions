import torch

def he_initialization(shape: tuple) -> torch.Tensor:
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:])
    std = torch.sqrt(2.0 / fan_in)
    return torch.randn(*shape) * std
