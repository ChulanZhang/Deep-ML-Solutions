"""Batched MHA with output projection. Related: Deep-ML #94, #107.
Mask: boolean / 0-1, True / 1 = allowed; broadcast to (B,H,S,S).
All-masked query rows produce zero context. No implicit causal mask.
"""
import torch
from torch import nn

def _attention(q, k, v, allowed=None):
    dtype = q.dtype
    q, k, v = (t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
               for t in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if allowed is not None:
        scores = scores.masked_fill(~allowed.bool(), float('-inf'))
    empty = torch.isneginf(scores).all(-1, keepdim=True)
    safe_scores = torch.where(empty, torch.zeros_like(scores), scores)
    probabilities = safe_scores.softmax(-1).masked_fill(empty, 0)
    return (probabilities @ v).to(dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads:
            raise ValueError('Positive n_heads must divide d_model.')
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        batch, length, width = x.shape
        split = lambda t: t.reshape(batch, length, self.n_heads, self.d_k).transpose(1, 2)
        q, k, v = (split(layer(x)) for layer in (self.q_proj, self.k_proj, self.v_proj))
        context = _attention(q, k, v, mask)
        return self.out_proj(context.transpose(1, 2).reshape(batch, length, width))
