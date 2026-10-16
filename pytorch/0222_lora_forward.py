"""Deep-ML #222: LoRA: Low-Rank Adaptation Forward Pass.
https://www.deep-ml.com/problems/222
Standalone PyTorch solution; public-example replay is not an official submission.
This problem names B as (in, rank), A as (rank, out). No detach or output rounding.
"""
import torch

def lora_forward(x, W, A, B, alpha=1.0):
    rank = A.shape[0]
    if rank <= 0 or B.shape[-1] != rank:
        raise ValueError('LoRA rank must be positive and consistent.')
    return x @ W + (alpha / rank) * ((x @ B) @ A)
