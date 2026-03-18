import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    """
    seq_len, d_k = Q.shape
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # apply mask
    mask = np.ones((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False
        
    scores[mask] = -1e9
    
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_scores)
    attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    
    out = np.dot(attn, V)
    return np.round(out, 4)
