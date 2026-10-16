"""Functional Adam update with explicit state. Related: #49.
Not AdamW: no decoupled weight decay. The returned tensors replace old state.
"""
import torch

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8):
    if t < 1:
        raise ValueError('Adam step t is one-indexed.')
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad.square()
    m_hat, v_hat = m / (1 - beta1**t), v / (1 - beta2**t)
    return param - lr * m_hat / (v_hat.sqrt() + eps), m, v
