import torch

def noisy_top_k_gating(X: torch.Tensor, W_g: torch.Tensor, W_noise: torch.Tensor, N: torch.Tensor, k: int) -> torch.Tensor:
    """
    Noisy Top-K Gating for MoE.
    X: (batch, features)
    W_g, W_noise: (features, experts)
    N: (batch, experts) Std normal noise
    """
    raw_scores = torch.matmul(X, W_g)
    noise_scaling = torch.matmul(X, W_noise)
    
    # Add noise: log(1 + exp(noise_scaling)) -> softplus
    # Avoid overflow in softplus
    softplus = np.log1p(torch.exp(-torch.abs(noise_scaling))) + torch.maximum(noise_scaling, 0)
    
    noisy_scores = raw_scores + softplus * N
    
    batch_size, num_experts = noisy_scores.shape
    gating_output = torch.zeros_like(noisy_scores)
    
    for i in range(batch_size):
        row = noisy_scores[i]
        # Get top k indices
        top_k_idx = np.argsort(row)[-k:]
        
        # Softmax over top k
        top_k_scores = row[top_k_idx]
        max_s = torch.amax(top_k_scores)
        exps = torch.exp(top_k_scores - max_s)
        probs = exps / torch.sum(exps)
        
        gating_output[i, top_k_idx] = probs
        
    return gating_output
