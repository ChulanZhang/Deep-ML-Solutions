import torch

def mhc_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, q_len, d_model = Q.shape
    _, kv_len, _ = K.shape
    head_dim = d_model // num_heads
    
    Q_r = Q.reshape(batch, q_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_r = K.reshape(batch, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_r = V.reshape(batch, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    scores = np.einsum('b h q d, b h k d -> b h q k', Q_r, K_r) / torch.sqrt(head_dim)
    
    max_scores = torch.amax(scores, dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - max_scores)
    attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    
    out = np.einsum('b h q k, b h k d -> b h q d', attn_weights, V_r)
    out = out.transpose(0, 2, 1, 3).reshape(batch, q_len, d_model)
    
    return torch.round(out, 4)
