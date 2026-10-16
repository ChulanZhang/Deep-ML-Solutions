"""Frozen-base LoRA module. Related: #222 has different A/B shape names.
Here A[rank,in], B[out,rank], output = base(x) + alpha/r * x A.T B.T.
Zero B is the conventional initialization, not the only possible LoRA initializer.
Load pretrained.weight explicitly when adapting an existing trained layer.
"""
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.):
        super().__init__()
        if r <= 0:
            raise ValueError('LoRA rank must be positive.')
        self.pretrained = nn.Linear(in_features, out_features, bias=False)
        self.pretrained.weight.requires_grad_(False)
        self.r, self.scaling = r, lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor):
        return self.pretrained(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)
