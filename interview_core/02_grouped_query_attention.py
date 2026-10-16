"""GQA reference module. Related: Deep-ML #391.
Stored K/V have fewer heads; repeat_interleave is a simple conceptual reference,
not an optimized implementation that avoids materializing repeated heads.
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

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        if min(d_model, n_q_heads, n_kv_heads) <= 0 or d_model % n_q_heads or n_q_heads % n_kv_heads:
            raise ValueError('d_model must divide into Q heads, and Q heads into KV groups.')
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.num_groups, self.d_k = n_q_heads // n_kv_heads, d_model // n_q_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        batch, length, width = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_q_heads, self.d_k).transpose(1, 2)
        k, v = (layer(x).reshape(batch, length, self.n_kv_heads, self.d_k).transpose(1, 2)
                for layer in (self.k_proj, self.v_proj))
        k, v = (t.repeat_interleave(self.num_groups, dim=1) for t in (k, v))
        context = _attention(q, k, v, mask)
        return self.out_proj(context.transpose(1, 2).reshape(batch, length, width))
