import numpy as np

def mhc_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_heads: int) -> np.ndarray:
    batch, q_len, d_model = Q.shape
    _, kv_len, _ = K.shape
    head_dim = d_model // num_heads
    
    Q_r = Q.reshape(batch, q_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_r = K.reshape(batch, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_r = V.reshape(batch, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    scores = np.einsum('b h q d, b h k d -> b h q k', Q_r, K_r) / np.sqrt(head_dim)
    
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    out = np.einsum('b h q k, b h k d -> b h q d', attn_weights, V_r)
    out = out.transpose(0, 2, 1, 3).reshape(batch, q_len, d_model)
    
    return np.round(out, 4)
