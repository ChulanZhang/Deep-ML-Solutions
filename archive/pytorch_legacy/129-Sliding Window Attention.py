import torch

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window attention.
    """
    seq_len, d_k = Q.shape
    scores = torch.matmul(Q, K.T) / torch.sqrt(d_k)
    
    # apply mask
    mask = torch.ones((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False
        
    scores[mask] = -1e9
    
    max_scores = torch.amax(scores, dim=-1, keepdim=True)
    exp_s = torch.exp(scores - max_scores)
    attn = exp_s / torch.sum(exp_s, dim=-1, keepdim=True)
    
    out = torch.matmul(attn, V)
    return torch.round(out, 4)
