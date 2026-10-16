"""Deep-ML #388: Sliding Window Attention.
https://www.deep-ml.com/problems/388
Standalone PyTorch solution; public-example replay is not an official submission.
Official #388 uses a bidirectional radius, NOT a causal decoder window.
"""
import torch

def _attention(Q, K, V, mask=None):
    # Accumulate low-precision scores in float32; preserve float64 inputs.
    dtype = Q.dtype
    q, k, v = (t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
               for t in (Q, K, V))
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float('-inf'))  # True = allowed
        else:
            scores = scores + mask
    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = torch.where(empty, torch.zeros_like(scores), scores)
    weights = torch.softmax(scores, dim=-1).masked_fill(empty, 0)
    return (weights @ v).to(dtype)

def compute_qkv(X, W_q, W_k, W_v):
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q, K, V):
    return _attention(Q, K, V)


def sliding_window_attention(Q, K, V, window_size):
    if window_size < 0:
        raise ValueError('window_size must be non-negative.')
    rows = torch.arange(Q.shape[-2], device=Q.device)[:, None]
    cols = torch.arange(K.shape[-2], device=Q.device)[None, :]
    mask = (rows - cols).abs() <= window_size
    return torch.round(_attention(Q, K, V, mask), decimals=4)
