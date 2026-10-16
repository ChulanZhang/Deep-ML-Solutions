import torch

def shared_expert_moe(x: torch.Tensor, shared_W: torch.Tensor, expert_Ws: list, gate_W: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(x, dtype=float)
    shared_W = torch.tensor(shared_W, dtype=float)
    gate_W = torch.tensor(gate_W, dtype=float)
    
    shared_out = torch.matmul(x, shared_W)
    
    logits = torch.matmul(x, gate_W)
    max_l = torch.amax(logits, dim=-1, keepdim=True)
    exp_l = torch.exp(logits - max_l)
    gates = exp_l / torch.sum(exp_l, dim=-1, keepdim=True)
    
    expert_out = torch.zeros_like(shared_out)
    for i, W in enumerate(expert_Ws):
        W = torch.tensor(W, dtype=float)
        expert_out += gates[:, i:i+1] * torch.matmul(x, W)
        
    return torch.round(shared_out + expert_out, 4)
