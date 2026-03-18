import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (Coefficient of Determination) value.
    
    Formula: R^2 = 1 - (SS_res / SS_tot)
    where:
        SS_res = sum( (y_true - y_pred)^2 )
        SS_tot = sum( (y_true - mean(y_true))^2 )
        
    Args:
        y_true: NumPy array of true values
        y_pred: NumPy array of predicted values
        
    Returns:
        R-squared value as a float
    """
    # Calculate Residual Sum of Squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate Total Sum of Squares (SS_tot)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Handle division by zero if all true values are the same
    if ss_tot == 0:
        return 0.0
        
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)
