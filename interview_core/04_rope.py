"""RoPE for x[B,S,H,D], using HALF-SPLIT pairs, not adjacent pairs.
Related: #381 uses adjacent pairs; these conventions are not interchangeable
without permuting the feature layout. start_pos supports cached decoding.
"""
import torch

def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0,
                         *, device=None, dtype=torch.float32):
    if dim <= 0 or dim % 2 or max_seq_len < 0 or base <= 0:
        raise ValueError('Need even positive dim, nonnegative length and positive base.')
    frequency = base ** (-torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = positions[:, None] * frequency
    return angles.cos(), angles.sin()

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0):
    length, width = x.shape[1], x.shape[-1]
    if width % 2 or start_pos < 0 or start_pos + length > cos.shape[0]:
        raise ValueError('Invalid feature dimension or position range.')
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    c = cos[start_pos:start_pos + length].to(work).unsqueeze(0).unsqueeze(2)
    s = sin[start_pos:start_pos + length].to(work).unsqueeze(0).unsqueeze(2)
    first, second = work.chunk(2, dim=-1)
    return torch.cat((first * c - second * s, second * c + first * s), dim=-1).to(x.dtype)
