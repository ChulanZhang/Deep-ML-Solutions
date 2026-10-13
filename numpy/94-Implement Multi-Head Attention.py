import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    """
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    return np.dot(attn_weights, V)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
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
        
    out = np.concatenate(head_outs, axis=-1)
    return out
