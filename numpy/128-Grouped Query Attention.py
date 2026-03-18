import numpy as np

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    Q: (B, S, H_q * D)
    K, V: (B, S, H_kv * D)
    """
    B, S, hidden_q = Q.shape
    _, _, hidden_kv = K.shape
    
    D = hidden_q // num_heads
    
    # Reshape to (B, S, Heads, D) -> (B, Heads, S, D)
    Q = Q.reshape(B, S, num_heads, D).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_kv_heads, D).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_kv_heads, D).transpose(0, 2, 1, 3)
    
    # Repeat K and V
    repeats = num_heads // num_kv_heads
    # K: (B, H_kv, S, D) -> repeat -> (B, H_kv, repeats, S, D) -> reshape -> (B, H_q, S, D)
    K_rep = np.repeat(K, repeats, axis=1)
    V_rep = np.repeat(V, repeats, axis=1)
    
    # Attention
    scores = np.matmul(Q, K_rep.transpose(0, 1, 3, 2)) / np.sqrt(D)
    
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    idx_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    out = np.matmul(idx_weights, V_rep)
    
    # Reshape back to (B, S, H_q * D)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, hidden_q)
    return out
