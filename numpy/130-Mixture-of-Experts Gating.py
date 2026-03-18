import numpy as np

def noisy_top_k_gating(X: np.ndarray, W_g: np.ndarray, W_noise: np.ndarray, N: np.ndarray, k: int) -> np.ndarray:
    """
    Noisy Top-K Gating for MoE.
    X: (batch, features)
    W_g, W_noise: (features, experts)
    N: (batch, experts) Std normal noise
    """
    raw_scores = np.dot(X, W_g)
    noise_scaling = np.dot(X, W_noise)
    
    # Add noise: log(1 + exp(noise_scaling)) -> softplus
    # Avoid overflow in softplus
    softplus = np.log1p(np.exp(-np.abs(noise_scaling))) + np.maximum(noise_scaling, 0)
    
    noisy_scores = raw_scores + softplus * N
    
    batch_size, num_experts = noisy_scores.shape
    gating_output = np.zeros_like(noisy_scores)
    
    for i in range(batch_size):
        row = noisy_scores[i]
        # Get top k indices
        top_k_idx = np.argsort(row)[-k:]
        
        # Softmax over top k
        top_k_scores = row[top_k_idx]
        max_s = np.max(top_k_scores)
        exps = np.exp(top_k_scores - max_s)
        probs = exps / np.sum(exps)
        
        gating_output[i, top_k_idx] = probs
        
    return gating_output
