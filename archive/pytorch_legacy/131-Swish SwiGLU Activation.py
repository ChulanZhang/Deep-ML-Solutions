import torch

def swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute SwiGLU 
    x: (B, 2d) -> Split to (B, d) and (B, d)
    """
    d = x.shape[1] // 2
    gate = x[:, :d]
    linear = x[:, d:]
    
    # Swish / SiLU
    sigmoid = 1.0 / (1.0 + torch.exp(-gate))
    swish = gate * sigmoid
    
    return swish * linear
