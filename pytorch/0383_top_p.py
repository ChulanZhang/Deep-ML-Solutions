"""Deep-ML #383: Top-p (Nucleus) Sampling.
https://www.deep-ml.com/problems/383
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def top_p_sampling(logits: list[float], p: float) -> torch.Tensor:
    if not 0 < p <= 1:
        raise ValueError('p must be in (0, 1].')
    x = torch.as_tensor(logits, dtype=torch.float64)
    probabilities = torch.softmax(x, dim=-1)
    order = torch.argsort(probabilities, descending=True, stable=True)
    ordered = probabilities[order]
    keep = (ordered.cumsum(-1) - ordered) < p
    keep[0] = True
    kept = ordered * keep
    return torch.zeros_like(x).scatter(0, order, kept / kept.sum())
