"""Deep-ML #223: QLoRA: Quantized Low-Rank Adaptation Forward Pass.
https://www.deep-ml.com/problems/223
Standalone PyTorch solution; public-example replay is not an official submission.
Exercise convention: W = quantized_W * scale + zero_point. Not a full NF4 QLoRA system.
"""
import torch

def qlora_forward(x, quantized_W, scale, zero_point, A, B, alpha=1.0):
    rank = A.shape[0]
    if rank <= 0 or B.shape[-1] != rank:
        raise ValueError('LoRA rank must be positive and consistent.')
    W = quantized_W.to(dtype=x.dtype, device=x.device) * scale + zero_point
    return x @ W + (alpha / rank) * ((x @ B) @ A)
