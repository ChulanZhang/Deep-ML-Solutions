import numpy as np

def prune_heads(W_q, W_k, W_v, W_o, heads_to_keep, num_heads):
    W_q = np.array(W_q)
    W_k = np.array(W_k)
    W_v = np.array(W_v)
    W_o = np.array(W_o)
    
    d_model = W_q.shape[0]
    head_dim = d_model // num_heads
    
    indices = []
    for h in heads_to_keep:
        indices.extend(range(h * head_dim, (h + 1) * head_dim))
        
    return W_q[:, indices], W_k[:, indices], W_v[:, indices], W_o[indices, :]
