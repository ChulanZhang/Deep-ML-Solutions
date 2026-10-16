import torch
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.
    """
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.T) / torch.sqrt(d_k)
    
    # numerically stable softmax
    scores_max = torch.amax(scores, dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)
    attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    
    return torch.matmul(attn_weights, V)

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.
    """
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    
    # Split into heads
    Q_split = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_split = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_split = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    head_outs = []
    for i in range(n_heads):
        out_i = self_attention(Q_split[i], K_split[i], V_split[i])
        head_outs.append(out_i)
        
    out = torch.cat(head_outs, dim=-1)
    return out
