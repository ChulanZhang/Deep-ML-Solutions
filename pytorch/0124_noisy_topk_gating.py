"""Deep-ML #124: Implement the Noisy Top-K Gating Function.
https://www.deep-ml.com/problems/124
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch
import torch.nn.functional as F

def noisy_topk_gating(X, W_g, W_noise, N, k):
    scores = X @ W_g + F.softplus(X @ W_noise) * N
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError('k must be between 1 and num_experts.')
    values, indices = torch.topk(scores, k, dim=-1)
    weights = torch.softmax(values, dim=-1)
    return torch.zeros_like(scores).scatter(-1, indices, weights)
