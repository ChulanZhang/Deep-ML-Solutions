import numpy as np

def multiquery_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    batch, num_heads, seq_len, head_dim = Q.shape
    
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(head_dim)
    
    max_s = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    
    out = np.einsum('bhqk,bhkd->bhqd', attn, V)
    return np.round(out, 4)
