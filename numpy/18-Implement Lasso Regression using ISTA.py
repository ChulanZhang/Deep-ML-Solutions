import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft-thresholding operator element-wise.
    S(w, lambda) = sign(w) * max(|w| - lambda, 0)
    """
    # np.maximum guarantees no negative values
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    Returns:
        tuple: (weights, bias)
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    
    for _ in range(max_iter):
        # 1. Gradient Step (Compute gradients of the smooth MSE part)
        # h(x) = X*w + b
        predictions = np.dot(X, weights) + bias
        error = predictions - y
        
        # Gradient of MSE w.r.t weights and bias
        # Note: Some definitions of MSE use 1/(2m) or 1/m, Deep-ML typically uses 1/m or 1/n_samples for MSE gradient.
        dw = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)
        
        # Temporarily update weights and bias using the gradient step
        weights_temp = weights - learning_rate * dw
        bias_new = bias - learning_rate * db
        
        # 2. Proximal Step (Apply soft-thresholding to the weights for L1 penalty)
        # We don't shrink the bias term, only the weights.
        weights_new = soft_threshold(weights_temp, learning_rate * alpha)
        
        # Check for convergence
        if np.max(np.abs(weights_new - weights)) < tol and np.abs(bias_new - bias) < tol:
            weights = weights_new
            bias = bias_new
            break
            
        weights = weights_new
        bias = bias_new
        
    return weights, bias
