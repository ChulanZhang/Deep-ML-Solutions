def token_sampling_pipeline(logits, temperature=1.0, top_k=0, top_p=1.0):
    import torch
    logits = torch.tensor(logits, dtype=float)
    
    # 1. Temperature
    if temperature > 0:
        logits = logits / temperature
        
    probs = torch.exp(logits - torch.amax(logits))
    probs /= torch.sum(probs)
    
    # 2. Top-K
    if top_k > 0:
        indices_to_remove = np.argsort(probs)[:-top_k]
        probs[indices_to_remove] = 0.0
        
    probs /= torch.sum(probs)
    
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
        
    sum_p = torch.sum(probs)
    if sum_p > 0:
        probs /= sum_p
        
    return torch.round(probs, 4)
