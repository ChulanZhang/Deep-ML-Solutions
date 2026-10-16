"""Deep-ML #391: Implement Grouped Query Attention (GQA).
https://www.deep-ml.com/problems/391
Standalone PyTorch solution; public-example replay is not an official submission.
Conceptual repeat_interleave reference; repetition does not demonstrate runtime KV bandwidth savings.
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


def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    batch, q_length, q_width = Q.shape
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError('num_heads must be a positive multiple of num_kv_heads.')
    if q_width % num_heads:
        raise ValueError('Query width must divide into equal heads.')
    dim = q_width // num_heads
    if K.shape[-1] != num_kv_heads * dim or V.shape != K.shape:
        raise ValueError('K/V dimensions do not match the head configuration.')
    q = Q.reshape(batch, q_length, num_heads, dim).transpose(1, 2)
    k, v = (t.reshape(batch, t.shape[1], num_kv_heads, dim).transpose(1, 2) for t in (K, V))
    repeats = num_heads // num_kv_heads
    k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
    return _attention(q, k, v).transpose(1, 2).reshape(batch, q_length, q_width)
