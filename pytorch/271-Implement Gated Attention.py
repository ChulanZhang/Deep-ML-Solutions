import torch

def gated_attention(x: torch.Tensor, w_gate: torch.Tensor, w_attn: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    w_gate = torch.tensor(w_gate, dtype=float)
    w_attn = torch.tensor(w_attn, dtype=float)
    
    gate_logits = torch.matmul(x, w_gate)
    gate = 1.0 / (1.0 + torch.exp(-gate_logits))
    
    attn_out = torch.matmul(x, w_attn)
    return torch.round(gate * attn_out, 4)
