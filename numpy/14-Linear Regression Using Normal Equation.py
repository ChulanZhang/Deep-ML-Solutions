import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """
    Perform linear regression using the normal equation.
    
    Formula: theta = (X^T * X)^-1 * X^T * y
    """
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Compute the coefficients using the pseudo-inverse for numerical stability
    X_T = X_arr.T
    theta_arr = np.linalg.pinv(X_T @ X_arr) @ X_T @ y_arr
    
    # Round to 4 decimal places
    theta = np.round(theta_arr, 4).tolist()
    
    # Replace any -0.0 with 0.0 for strict equality matching
    return [0.0 if x == -0.0 else x for x in theta]
