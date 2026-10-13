import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    x_arr = torch.tensor(x, dtype=float)
    d = x_arr.shape[-1] // 2
    x1, x2 = x_arr[..., :d], x_arr[..., d:]
    
    # Swish = x1 * sigmoid(x1)
    swish = x1 / (1.0 + torch.exp(-x1))
    
    return torch.round(swish * x2, 4)
