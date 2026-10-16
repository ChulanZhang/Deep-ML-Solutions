"""Deep-ML #407: QK-Norm (Query-Key Normalization).
https://www.deep-ml.com/problems/407
Standalone PyTorch solution; public-example replay is not an official submission.
After L2 normalization, scale by 1/temperature; do not add another sqrt(d) divisor.
"""
import torch
import torch.nn.functional as F

def qk_norm_attention(Q, K, V, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive.')
    q = F.normalize(Q, dim=-1, eps=1e-8)
    k = F.normalize(K, dim=-1, eps=1e-8)
    weights = torch.softmax((q @ k.T) / temperature, dim=-1)
    return weights @ V, weights
