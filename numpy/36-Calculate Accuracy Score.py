import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score of a model's predictions.
    
    Formula: Accuracy = (True Positives + True Negatives) / Total Population
             or simply: Number of Correct Predictions / Total Predictions
             
    Args:
        y_true: 1D numpy array of actual labels
        y_pred: 1D numpy array of predicted labels
        
    Returns:
        Accuracy score as a float
    """
    # Simply count where predictions match the true labels exactly and divide by total
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    if total_predictions == 0:
        return 0.0
        
    return float(correct_predictions / total_predictions)
