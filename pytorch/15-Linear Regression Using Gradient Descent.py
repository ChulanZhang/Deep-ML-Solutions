import torch

def linear_regression_gradient_descent(X: torch.Tensor, y: torch.Tensor, alpha: float, iterations: int) -> torch.Tensor:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations

    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    
    # 1. Initialize all weights to zero
    theta = torch.zeros(n)
    
    # 2. Gradient Descent Loop
    for _ in range(iterations):
        # a. Make predictions using current weights
        # h_theta(x) = X * theta (Dot product)
        predictions = torch.matmul(X, theta)
        
        # b. Calculate the error
        error = predictions - y
        
        # c. Calculate the gradient
        # The partial derivative of MSE Loss w.r.t theta_j is:
        # (1/m) * sum((h_theta(x) - y) * x_j)
        # Using vectorization: X.T dot error computes this for all j simultaneously
        gradient = (1 / m) * torch.matmul(X.T, error)
        
        # d. Update weights
        theta = theta - alpha * gradient
        
    return theta
