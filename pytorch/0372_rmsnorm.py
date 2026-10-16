"""Deep-ML #372: Implement RMSNorm (Root Mean Square Layer Normalization).
https://www.deep-ml.com/problems/372
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def rmsnorm(x, g, eps=1e-5):
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms_inverse = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    return (work * rms_inverse * g).to(x.dtype)
