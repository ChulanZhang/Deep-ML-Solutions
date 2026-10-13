import torch

def ptx_loss(logits: torch.Tensor, labels: torch.Tensor, ptx_coef: float = 1.0) -> float:
    logits = torch.tensor(logits, dtype=float)
    labels = torch.tensor(labels, dtype=int)
    
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    exp_logits = torch.exp(logits - max_logits)
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    if logits.ndim == 2:
        # (batch, vocab)
        batch_idx = torch.arange(labels.shape[0])
        true_probs = probs[batch_idx, labels]
    else:
        batch_idx, seq_idx = np.indices(labels.shape)
        true_probs = probs[batch_idx, seq_idx, labels]
        
    ce_loss = -torch.log(torch.clamp(true_probs, 1e-10, 1.0))
    return float(ptx_coef * torch.mean(ce_loss))
