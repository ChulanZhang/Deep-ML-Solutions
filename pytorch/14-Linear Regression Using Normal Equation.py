import torch

def linear_regression_normal_equation(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Perform linear regression using the normal equation.
    
    Formula: theta = (X^T * X)^-1 * X^T * y
    """
    X_arr = torch.tensor(X)
    y_arr = torch.tensor(y)
    
    # Compute the coefficients using the pseudo-inverse for numerical stability
    X_T = X_arr.T
    theta_arr = np.linalg.pinv(X_T @ X_arr) @ X_T @ y_arr
    
    # Round to 4 decimal places
    theta = torch.round(theta_arr, 4).tolist()
    
    # Replace any -0.0 with 0.0 for strict equality matching
    return [0.0 if x == -0.0 else x for x in theta]
