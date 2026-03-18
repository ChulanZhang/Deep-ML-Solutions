import numpy as np

def top_p_sampling(probs: np.ndarray, p: float) -> np.ndarray:
    probs = np.array(probs, dtype=float)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    cumulative_probs = np.cumsum(sorted_probs)
    
    to_remove = cumulative_probs > p
    to_remove[1:] = to_remove[:-1].copy()
    to_remove[0] = False
    
    sorted_probs[to_remove] = 0.0
    sum_probs = np.sum(sorted_probs)
    if sum_probs > 0:
        sorted_probs /= sum_probs
        
    out_probs = np.zeros_like(probs)
    out_probs[sorted_indices] = sorted_probs
    
    return np.round(out_probs, 4)
