import torch

def engram_gating(x: torch.Tensor, context: torch.Tensor, W_x: torch.Tensor, W_c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    c = torch.tensor(context, dtype=float)
    
    gate_logits = torch.matmul(x, W_x) + torch.matmul(c, W_c) + b
    gate = 1.0 / (1.0 + torch.exp(-gate_logits))
    
    return torch.round(gate * x, 4)
