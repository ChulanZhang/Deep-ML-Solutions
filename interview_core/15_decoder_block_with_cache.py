"""Composition exercise, NOT a separately numbered Deep-ML problem.
Pre-norm RMSNorm + causal GQA + adjacent-pair RoPE + KV cache + SwiGLU.
Related exercises: #109/#372, #391, #381, #376, #156/#178, #408.
No dropout, padding mask, cache eviction or fused kernels in this reference.
"""
import torch
from torch import nn
from torch.nn import functional as F

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, hidden_dim: int,
                 eps: float = 1e-6, rope_base: float = 10000.):
        super().__init__()
        if min(d_model, n_q_heads, n_kv_heads) <= 0 or d_model % n_q_heads or n_q_heads % n_kv_heads:
            raise ValueError('Invalid model width or head counts.')
        self.head_dim = d_model // n_q_heads
        if self.head_dim % 2:
            raise ValueError('RoPE requires an even head dimension.')
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.eps, self.rope_base = eps, rope_base
        self.norm1 = nn.Parameter(torch.ones(d_model))
        self.norm2 = nn.Parameter(torch.ones(d_model))
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def _norm(self, x, weight):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps) * weight).to(x.dtype)

    def _rope(self, x, positions):
        # x: (B,H,S,D); rotate adjacent features at absolute token positions.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        freq = self.rope_base ** (-torch.arange(0, self.head_dim, 2, device=x.device, dtype=work.dtype) / self.head_dim)
        angle = positions.to(work.dtype)[:, None] * freq
        even, odd = work[..., 0::2], work[..., 1::2]
        c, s = angle.cos(), angle.sin()
        return torch.stack((even*c-odd*s, even*s+odd*c), -1).flatten(-2).to(x.dtype)

    def forward(self, x: torch.Tensor, cache: tuple | None = None):
        batch, length, width = x.shape
        past = 0 if cache is None else cache[0].shape[-2]
        positions = past + torch.arange(length, device=x.device)
        normalized = self._norm(x, self.norm1)
        split = lambda t, heads: t.reshape(batch, length, heads, self.head_dim).transpose(1, 2)
        q = self._rope(split(self.q_proj(normalized), self.n_q_heads), positions)
        k = self._rope(split(self.k_proj(normalized), self.n_kv_heads), positions)
        v = split(self.v_proj(normalized), self.n_kv_heads)
        if cache is not None:
            # Cached keys are ALREADY rotated. Do not rotate them a second time.
            k, v = torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2)
        new_cache = (k, v)  # Store H_kv heads, NOT expanded H_q heads.
        groups = self.n_q_heads // self.n_kv_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        q, k, v = (t.float() if t.dtype in (torch.float16, torch.bfloat16) else t for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        keys = torch.arange(past + length, device=x.device)
        allowed = keys[None, :] <= positions[:, None]
        weights = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
        context = (weights @ v).to(x.dtype).transpose(1, 2).reshape(batch, length, width)
        h = x + self.out_proj(context)
        normalized = self._norm(h, self.norm2)
        out = h + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return out, new_cache
