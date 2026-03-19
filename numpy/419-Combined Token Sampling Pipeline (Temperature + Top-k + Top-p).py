def token_sampling_pipeline(logits, temperature=1.0, top_k=0, top_p=1.0):
    import numpy as np
    logits = np.array(logits, dtype=float)
    
    # 1. Temperature
    if temperature > 0:
        logits = logits / temperature
        
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)
    
    # 2. Top-K
    if top_k > 0:
        indices_to_remove = np.argsort(probs)[:-top_k]
        probs[indices_to_remove] = 0.0
        
    probs /= np.sum(probs)
    
    # 3. Top-P
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        
        to_remove = cumulative > top_p
        to_remove[1:] = to_remove[:-1].copy()
        to_remove[0] = False
        
        sorted_probs[to_remove] = 0.0
        probs[sorted_idx] = sorted_probs
        
    sum_p = np.sum(probs)
    if sum_p > 0:
        probs /= sum_p
        
    return np.round(probs, 4)
