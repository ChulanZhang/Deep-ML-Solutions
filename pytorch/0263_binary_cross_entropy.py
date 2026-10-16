"""Deep-ML #263: Implement Binary Cross-Entropy Loss.
https://www.deep-ml.com/problems/263
Standalone PyTorch solution; public-example replay is not an official submission.
Exercise returns a Python float. Use a Tensor-valued loss during training.
"""
import torch

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    if not 0 < epsilon < 0.5:
        raise ValueError('epsilon must be in (0, 0.5).')
    # float64 makes 1-epsilon distinguishable from 1 for the default epsilon.
    p = y_pred.to(torch.float64).clamp(epsilon, 1 - epsilon)
    y = y_true.to(torch.float64)
    return (-(y * p.log() + (1 - y) * torch.log1p(-p))).mean().item()
