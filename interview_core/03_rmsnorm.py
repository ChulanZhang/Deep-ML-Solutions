"""RMSNorm with fp32 accumulation for fp16/bfloat16 inputs. Related: #372.
Unlike LayerNorm, it does not subtract the mean. No zero-mean assumption is needed.
"""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)
