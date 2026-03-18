import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    """
    Compute the log-softmax of an array of scores.
    Uses numerical stability tricks by subtracting the maximum value.
    
    Formula: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    
    Args:
        scores: List or numpy array of float values
        
    Returns:
        Numpy array of log probabilities, rounded to 4 decimals
    """
    # Convert to numpy array for vectorized operations
    x = np.array(scores, dtype=np.float64)
    
    if x.size == 0:
        return x
        
    # 1. Find the maximum element for numerical stability
    x_max = np.max(x)
    
    # 2. Subtract max to prevent overflow in exp()
    shifted_x = x - x_max
    
    # 3. Compute log of the sum of exponentials: log(sum(e^(x - max)))
    log_sum_exp = np.log(np.sum(np.exp(shifted_x)))
    
    # 4. Compute final log softmax: (x - max) - log_sum_exp
    result = shifted_x - log_sum_exp
    
    return np.round(result, 4)
