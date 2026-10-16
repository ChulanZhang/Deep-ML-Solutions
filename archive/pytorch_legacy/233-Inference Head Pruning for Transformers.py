import torch

def prune_heads(W_q, W_k, W_v, W_o, heads_to_keep, num_heads):
    W_q = torch.tensor(W_q)
    W_k = torch.tensor(W_k)
    W_v = torch.tensor(W_v)
    W_o = torch.tensor(W_o)
    
    d_model = W_q.shape[0]
    head_dim = d_model // num_heads
    
    indices = []
    for h in heads_to_keep:
        indices.extend(range(h * head_dim, (h + 1) * head_dim))
        
    return W_q[:, indices], W_k[:, indices], W_v[:, indices], W_o[indices, :]
