import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.T) / torch.sqrt(d_k)
    
    # Apply mask (assume 0 means masked out)
    # Using a very large negative number for masking
    scores = np.where(mask == 0, -1e9, scores)
    
    scores_max = torch.amax(scores, dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)
    attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    
    return torch.matmul(attn_weights, V)
