import torch

def soft_threshold(w: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply soft-thresholding operator element-wise.
    S(w, lambda) = sign(w) * max(|w| - lambda, 0)
    """
    # np.maximum guarantees no negative values
    return np.sign(w) * torch.maximum(torch.abs(w) - threshold, 0.0)

def l1_regularization_gradient_descent(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    Returns:
        tuple: (weights, bias)
    """
    m, n = X.shape
    weights = torch.zeros(n)
    bias = 0.0
    
    for _ in range(max_iter):
        # 1. Gradient Step (Compute gradients of the smooth MSE part)
        # h(x) = X*w + b
        predictions = torch.matmul(X, weights) + bias
        error = predictions - y
        
        # Gradient of MSE w.r.t weights and bias
        # Note: Some definitions of MSE use 1/(2m) or 1/m, Deep-ML typically uses 1/m or 1/n_samples for MSE gradient.
        dw = (1/m) * torch.matmul(X.T, error)
        db = (1/m) * torch.sum(error)
        
        # Temporarily update weights and bias using the gradient step
        weights_temp = weights - learning_rate * dw
        bias_new = bias - learning_rate * db
        
        # 2. Proximal Step (Apply soft-thresholding to the weights for L1 penalty)
        # We don't shrink the bias term, only the weights.
        weights_new = soft_threshold(weights_temp, learning_rate * alpha)
        
        # Check for convergence
        if torch.amax(torch.abs(weights_new - weights)) < tol and torch.abs(bias_new - bias) < tol:
            weights = weights_new
            bias = bias_new
            break
            
        weights = weights_new
        bias = bias_new
        
    return weights, bias
