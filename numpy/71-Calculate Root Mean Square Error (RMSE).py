import numpy as np

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).
    
    Formula: RMSE = sqrt( mean( (y_true - y_pred)^2 ) )
    
    Args:
        y_true: NumPy array of true values
        y_pred: NumPy array of predicted values
        
    Returns:
        RMSE value rounded to three decimal places (float)
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_res = np.sqrt(mse)
    return round(float(rmse_res), 3)
