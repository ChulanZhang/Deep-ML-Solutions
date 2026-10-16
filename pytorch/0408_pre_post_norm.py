"""Deep-ML #408: Pre-Norm vs Post-Norm Transformer Block.
https://www.deep-ml.com/problems/408
Standalone PyTorch solution; public-example replay is not an official submission.
Two linear stand-in sublayers, not a complete Transformer decoder.
"""
import torch
import torch.nn.functional as F

def transformer_block(x, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, mode, eps=1e-5):
    def ln(value, gamma, beta):
        return F.layer_norm(value, (value.shape[-1],), gamma, beta, eps)
    if mode == 'pre_norm':
        h = x + ln(x, gamma1, beta1) @ W1 + b1
        return h + ln(h, gamma2, beta2) @ W2 + b2
    if mode == 'post_norm':
        h = ln(x + x @ W1 + b1, gamma1, beta1)
        return ln(h + h @ W2 + b2, gamma2, beta2)
    raise ValueError('mode must be pre_norm or post_norm.')
