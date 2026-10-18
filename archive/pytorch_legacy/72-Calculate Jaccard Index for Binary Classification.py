import torch

def jaccard_index(y_true, y_pred):
    """
    Calculate the Jaccard Index for binary classification.
    
    Formula: Jaccard Index = Intersection / Union
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    intersection = torch.sum((y_true == 1) & (y_pred == 1))
    union = torch.sum((y_true == 1) | (y_pred == 1))
    
    if union == 0:
        return 1.0 if np.all(y_true == y_pred) else 0.0
        
    result = float(intersection / union)
    return round(result, 3)
