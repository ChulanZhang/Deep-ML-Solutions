"""Deep-ML #125: Implement a Sparse Mixture of Experts Layer.
https://www.deep-ml.com/problems/125
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def moe(x, We, Wg, n_experts, top_k):
    if We.shape[0] != n_experts or Wg.shape[-1] != n_experts:
        raise ValueError('Expert count does not match weights.')
    if not 1 <= top_k <= n_experts:
        raise ValueError('top_k must be between 1 and n_experts.')
    logits = x @ Wg
    values, indices = torch.topk(logits, top_k, dim=-1)
    weights = torch.softmax(values, dim=-1)
    # Gather only selected expert matrices, then combine selected outputs.
    selected = We[indices]  # (..., k, d_in, d_out)
    outputs = torch.einsum('...d,...kdf->...kf', x, selected)
    return (outputs * weights.unsqueeze(-1)).sum(dim=-2)
