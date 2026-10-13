import torch

def infonce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> float:
    z_i = torch.tensor(z_i, dtype=float)
    z_j = torch.tensor(z_j, dtype=float)
    
    z_i = z_i / (np.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-8)
    z_j = z_j / (np.linalg.norm(z_j, dim=-1, keepdim=True) + 1e-8)
    
    logits = torch.matmul(z_i, z_j.T) / temperature
    
    batch_size = z_i.shape[0]
    labels = torch.arange(batch_size)
    
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    exp_logits = torch.exp(logits - max_logits)
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    true_probs = probs[torch.arange(batch_size), labels]
    loss = -torch.mean(torch.log(torch.clamp(true_probs, 1e-10, 1.0)))
    return float(torch.round(loss, 4))
