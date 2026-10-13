import torch

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
    y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values

    Returns:
    float: Mean Absolute Error
    """
    # MAE = mean(abs(y_true - y_pred))
    error = torch.abs(y_true - y_pred)
    return float(torch.mean(error))
