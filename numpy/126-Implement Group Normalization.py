import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    """
    Implement Group Normalization.
    X: (N, C, H, W)
    """
    N, C, H, W = X.shape
    G = num_groups
    
    X_reshaped = X.reshape(N, G, C // G, H, W)
    
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    
    X_norm = (X_reshaped - mean) / np.sqrt(var + epsilon)
    X_norm = X_norm.reshape(N, C, H, W)
    
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    
    return gamma * X_norm + beta
