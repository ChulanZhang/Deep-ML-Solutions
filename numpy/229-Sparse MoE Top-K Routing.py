import numpy as np

def topk_routing(x: np.ndarray, w_gate: np.ndarray, k: int) -> tuple:
    x_arr = np.array(x, dtype=float)
    w_gate_arr = np.array(w_gate, dtype=float)
    
    # Calculate initial scores
    scores = np.dot(x_arr, w_gate_arr)
    
    # Sort backwards to get top-k
    sorted_idx = np.argsort(scores, axis=-1)
    top_indices = sorted_idx[:, -k:]    # The last k elements are the largest
    top_indices = top_indices[:, ::-1]  # Reverse them so biggest is first
    
    # Extract scores
    top_scores = np.take_along_axis(scores, top_indices, axis=1)
    
    # Compute Softmax strictly over the top_scores
    max_scores = np.max(top_scores, axis=1, keepdims=True)
    exp_scores = np.exp(top_scores - max_scores)
    
    gating_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return np.round(gating_weights, 4), top_indices
