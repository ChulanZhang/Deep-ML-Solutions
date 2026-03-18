import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # 1. Compute linear combination
    z = np.dot(X, weights) + bias
    
    # 2. Apply sigmoid function
    probabilities = 1.0 / (1.0 + np.exp(-z))
    
    # 3. Apply threshold of 0.5
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions
