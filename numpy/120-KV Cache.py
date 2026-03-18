import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple = None) -> tuple:
    """
    Perform a single attention step with KV caching.
    """
    q_new = np.dot(x_new, W_Q)
    k_new = np.dot(x_new, W_K)
    v_new = np.dot(x_new, W_V)
    
    if cache is not None:
        K_cache, V_cache = cache
        K_updated = np.vstack([K_cache, k_new])
        V_updated = np.vstack([V_cache, v_new])
    else:
        K_updated = k_new[np.newaxis, :]
        V_updated = v_new[np.newaxis, :]
        
    d_k = q_new.shape[0]
    
    scores = np.dot(K_updated, q_new) / np.sqrt(d_k)
    max_score = np.max(scores)
    exp_s = np.exp(scores - max_score)
    attn_weights = exp_s / np.sum(exp_s)
    
    output = np.dot(attn_weights, V_updated)
    
    return output, (K_updated, V_updated)
