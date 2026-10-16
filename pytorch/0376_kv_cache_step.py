"""Deep-ML #376: KV Cache for Efficient Autoregressive Attention.
https://www.deep-ml.com/problems/376
Standalone PyTorch solution; public-example replay is not an official submission.
Single-token step only; all cached tokens are already causally visible.
"""
import torch

def kv_cache_attention_step(x_new, W_Q, W_K, W_V, cache):
    q, k, v = x_new @ W_Q, x_new @ W_K, x_new @ W_V
    k, v = k.unsqueeze(0), v.unsqueeze(0)
    if cache is not None:
        k = torch.cat((cache[0], k), dim=0)
        v = torch.cat((cache[1], v), dim=0)
    q_work, k_work, v_work = (t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
                             for t in (q, k, v))
    scores = (k_work @ q_work) / (q.shape[-1] ** 0.5)
    out = (torch.softmax(scores, dim=-1) @ v_work).to(v.dtype)
    return out, (k, v)
