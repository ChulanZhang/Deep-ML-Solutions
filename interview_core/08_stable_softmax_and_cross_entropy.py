"""Differentiable stable softmax and cross entropy. Related: #23, #39.
Logits are (..., vocabulary), targets are (...). No output rounding / tolist().
"""
import torch

def stable_softmax(logits: torch.Tensor, dim: int = -1):
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    exps = (work - work.amax(dim=dim, keepdim=True)).exp()
    return (exps / exps.sum(dim=dim, keepdim=True)).to(logits.dtype)

def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_normalizer = torch.logsumexp(work, dim=-1)
    selected = work.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - selected).mean()
