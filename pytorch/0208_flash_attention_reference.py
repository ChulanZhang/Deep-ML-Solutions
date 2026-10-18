"""Deep-ML #208: Flash Attention v1 - Forward Pass.
https://www.deep-ml.com/problems/208
Standalone PyTorch solution; public-example replay is not an official submission.
Mathematical blockwise online-softmax reference, NOT a fused GPU FlashAttention kernel.
"""
import torch

def flash_attention_forward(Q, K, V, block_size=2):
    if block_size <= 0 or K.shape[0] == 0:
        raise ValueError('block_size and key sequence length must be positive.')
    dtype = Q.dtype
    q, k, v = (t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
               for t in (Q, K, V))
    if q.shape[0] == 0:
        return V.new_empty((0, V.shape[-1]))
    chunks = []
    for start in range(0, q.shape[0], block_size):
        qb = q[start:start + block_size]
        maximum = qb.new_full((qb.shape[0], 1), float('-inf'))
        denominator = qb.new_zeros((qb.shape[0], 1))
        numerator = qb.new_zeros((qb.shape[0], v.shape[-1]))
        for key_start in range(0, k.shape[0], block_size):
            kb, vb = k[key_start:key_start + block_size], v[key_start:key_start + block_size]
            scores = (qb @ kb.T) / (q.shape[-1] ** 0.5)
            new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            correction = torch.exp(maximum - new_maximum)
            probabilities = torch.exp(scores - new_maximum)
            numerator = correction * numerator + probabilities @ vb
            denominator = correction * denominator + probabilities.sum(-1, keepdim=True)
            maximum = new_maximum
        chunks.append(numerator / denominator)
    return torch.cat(chunks, dim=0).to(dtype)
