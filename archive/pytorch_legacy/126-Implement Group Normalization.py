import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Implement Group Normalization.
    X: (N, C, H, W)
    """
    N, C, H, W = X.shape
    G = num_groups
    
    X_reshaped = X.reshape(N, G, C // G, H, W)
    
    mean = torch.mean(X_reshaped, dim=(2, 3, 4), keepdim=True)
    var = torch.var(X_reshaped, dim=(2, 3, 4), keepdim=True)
    
    X_norm = (X_reshaped - mean) / torch.sqrt(var + epsilon)
    X_norm = X_norm.reshape(N, C, H, W)
    
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    
    return gamma * X_norm + beta
