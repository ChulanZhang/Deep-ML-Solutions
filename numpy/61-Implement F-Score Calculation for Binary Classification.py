import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate the F-beta score, combining Precision and Recall.
    
    Formula: F_beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)
    
    Args:
        y_true: Numpy array of true binary labels (1 or 0)
        y_pred: Numpy array of predicted binary labels (1 or 0)
        beta: Float adjusting the importance of Precision vs Recall
        
    Returns:
        F-score as a float rounded to 3 decimal places. Returns 0.0 if undefined mathematically.
    """
    # 1. Compute Precision
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    prec_denom = tp + fp
    precision = float(tp / prec_denom) if prec_denom > 0 else 0.0
    
    # 2. Compute Recall
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    rec_denom = tp + fn
    recall = float(tp / rec_denom) if rec_denom > 0 else 0.0
    
    # 3. Handle edge cases where both are 0
    if precision == 0.0 and recall == 0.0:
        return 0.0
        
    # 4. Calculate F-beta score
    beta_sq = float(beta) ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    
    return round(f_beta, 3)
