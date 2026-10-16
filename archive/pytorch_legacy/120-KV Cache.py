import torch

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple = None) -> tuple:
    """
    Perform a single attention step with KV caching.
    """
    q_new = torch.matmul(x_new, W_Q)
    k_new = torch.matmul(x_new, W_K)
    v_new = torch.matmul(x_new, W_V)
    
    if cache is not None:
        K_cache, V_cache = cache
        K_updated = torch.vstack([K_cache, k_new])
        V_updated = torch.vstack([V_cache, v_new])
    else:
        K_updated = k_new[np.newaxis, :]
        V_updated = v_new[np.newaxis, :]
        
    d_k = q_new.shape[0]
    
    scores = torch.matmul(K_updated, q_new) / torch.sqrt(d_k)
    max_score = torch.amax(scores)
    exp_s = torch.exp(scores - max_score)
    attn_weights = exp_s / torch.sum(exp_s)
    
    output = torch.matmul(attn_weights, V_updated)
    
    return output, (K_updated, V_updated)
