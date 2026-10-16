"""Deep-ML #107: Implement Masked Self-Attention.
https://www.deep-ml.com/problems/107
Standalone PyTorch solution; public-example replay is not an official submission.
Official masks are additive (0 / -inf); bool masks also work: True = allowed.
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


def masked_attention(Q, K, V, mask):
    return _attention(Q, K, V, mask)
