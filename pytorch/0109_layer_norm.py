"""Deep-ML #109: Implement Layer Normalization for Sequence Data.
https://www.deep-ml.com/problems/109
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def layer_normalization(X, gamma, beta, epsilon=1e-5):
    x = X.float() if X.dtype in (torch.float16, torch.bfloat16) else X
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).square().mean(dim=-1, keepdim=True)
    out = (x - mean) * torch.rsqrt(var + epsilon)
    return (out * gamma + beta).to(X.dtype)
