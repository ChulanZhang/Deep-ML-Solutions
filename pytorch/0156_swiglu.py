"""Deep-ML #156: Implement SwiGLU activation function.
https://www.deep-ml.com/problems/156
Standalone PyTorch solution; public-example replay is not an official submission.
Problem #156 uses first_half * SiLU(second_half), not the reverse.
"""
import torch
import torch.nn.functional as F

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError('The last dimension must be even.')
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    value, gate = x.chunk(2, dim=-1)
    return value * F.silu(gate)
