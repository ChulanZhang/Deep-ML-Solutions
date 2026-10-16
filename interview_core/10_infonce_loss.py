"""Symmetric CLIP-style InfoNCE for aligned feature batches (N,D).
This is NOT #384's SimCLR objective: #384 also uses same-view negatives.
Related concept: #384; do not copy this function into that exercise unchanged.
"""
import torch
from torch.nn import functional as F

def infonce_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float = 0.07):
    if temperature <= 0 or features_a.shape != features_b.shape:
        raise ValueError('Need positive temperature and matching paired batches.')
    a, b = F.normalize(features_a, dim=-1), F.normalize(features_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
