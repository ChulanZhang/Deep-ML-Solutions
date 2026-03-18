import numpy as np

def jaccard_index(y_true, y_pred):
    """
    Calculate the Jaccard Index for binary classification.
    
    Formula: Jaccard Index = Intersection / Union
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    
    if union == 0:
        return 1.0 if np.all(y_true == y_pred) else 0.0
        
    result = float(intersection / union)
    return round(result, 3)
