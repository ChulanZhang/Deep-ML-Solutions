import torch

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Formula: Recall = True Positives / (True Positives + False Negatives)
             (How many relevant items were selected out of all relevant items)
             
    Args:
        y_true: numpy array or list of actual binary labels
        y_pred: numpy array or list of predicted binary labels
        
    Returns:
        Recall as a float. Returns 0.0 if the denominator is 0.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
        
    # True Positives: Actual is 1 and Predicted is 1
    tp = torch.sum((y_true == 1) & (y_pred == 1))
    
    # False Negatives: Actual is 1 but Predicted is 0
    fn = torch.sum((y_true == 1) & (y_pred == 0))
    
    denominator = tp + fn
    
    if denominator == 0:
        return 0.0
        
    return float(tp / denominator)
