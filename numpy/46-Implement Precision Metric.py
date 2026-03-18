import numpy as np

def precision(y_true, y_pred):
    """
    Calculate the precision metric for binary classification.
    
    Formula: Precision = True Positives / (True Positives + False Positives)
    
    Args:
        y_true: numpy array of actual binary labels
        y_pred: numpy array of predicted binary labels
        
    Returns:
        Precision as a float. Returns 0.0 if the denominator is 0.
    """
    # True Positives: Actual is 1 and Predicted is 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # False Positives: Actual is 0 and Predicted is 1
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    denominator = tp + fp
    
    if denominator == 0:
        return 0.0
        
    return float(tp / denominator)
