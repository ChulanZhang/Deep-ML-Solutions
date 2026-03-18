import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.
    """
    N, C, H, W = x.shape
    out = np.zeros_like(x, dtype=float)
    half_n = n // 2
    
    x_sq = x ** 2
    
    for i in range(C):
        start_j = max(0, i - half_n)
        end_j = min(C, i + half_n + 1)
        
        # sum over channels for each spatial coordinate
        sq_sum = np.sum(x_sq[:, start_j:end_j, :, :], axis=1)
        
        scale = k + alpha * sq_sum
        out[:, i, :, :] = x[:, i, :, :] / (scale ** beta)
        
    return out
