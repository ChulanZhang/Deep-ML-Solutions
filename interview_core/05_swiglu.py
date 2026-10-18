"""Trainable SwiGLU FFN. Related: activation-only #156, FFN exercise #178.
Output = down(SiLU(gate(x)) * up(x)). Hidden size is an architecture choice.
"""
import torch
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(8 * d_model / 3) if hidden_dim is None else hidden_dim
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
