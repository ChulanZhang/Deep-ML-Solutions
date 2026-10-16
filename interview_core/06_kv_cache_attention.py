"""Causal MHA for prefill, single-token decode and multi-token chunks.
Related: #376 only asks for a single-head, single-token cache step.
Cache shape: (B,H,past_length,head_dim). No RoPE is applied in this module.
The returned cache retains autograd history; use no_grad() for inference.
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

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads:
            raise ValueError('Positive n_heads must divide d_model.')
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: tuple | None = None):
        batch, length, width = x.shape
        split = lambda t: t.reshape(batch, length, self.n_heads, self.d_k).transpose(1, 2)
        q, k, v = (split(layer(x)) for layer in (self.q_proj, self.k_proj, self.v_proj))
        past = 0 if kv_cache is None else kv_cache[0].shape[-2]
        if kv_cache is not None:
            k, v = torch.cat((kv_cache[0], k), -2), torch.cat((kv_cache[1], v), -2)
        # Absolute positions are essential when queries cover a cached chunk.
        q_positions = past + torch.arange(length, device=x.device)
        k_positions = torch.arange(past + length, device=x.device)
        allowed = k_positions[None, :] <= q_positions[:, None]
        context = _attention(q, k, v, allowed)
        out = context.transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(out), (k, v)
