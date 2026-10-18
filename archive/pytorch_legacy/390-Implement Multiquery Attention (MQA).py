import torch

def multiquery_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = Q.shape
    
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / torch.sqrt(head_dim)
    
    max_s = torch.amax(scores, dim=-1, keepdim=True)
    exp_s = torch.exp(scores - max_s)
    attn = exp_s / torch.sum(exp_s, dim=-1, keepdim=True)
    
    out = np.einsum('bhqk,bhkd->bhqd', attn, V)
    return torch.round(out, 4)
