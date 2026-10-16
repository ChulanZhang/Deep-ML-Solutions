"""Deep-ML #381: Rotary Positional Embeddings (RoPE).
https://www.deep-ml.com/problems/381
Standalone PyTorch solution; public-example replay is not an official submission.
Adjacent-pair convention: (x0,x1), (x2,x3), ...; rotate Q/K, not V.
"""
import torch

def apply_rope(x, positions, base=10000):
    dim = x.shape[-1]
    if dim % 2 or base <= 0:
        raise ValueError('RoPE requires an even dimension and positive base.')
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    positions = torch.as_tensor(positions, device=x.device, dtype=work.dtype)
    inverse_freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=work.dtype) / dim)
    angles = positions[..., None] * inverse_freq
    even, odd = work[..., 0::2], work[..., 1::2]
    cosine, sine = angles.cos(), angles.sin()
    out = torch.stack((even * cosine - odd * sine,
                       even * sine + odd * cosine), dim=-1)
    return out.flatten(-2).to(x.dtype)
