import torch

def predict_logistic(X: torch.Tensor, weights: torch.Tensor, bias: float) -> torch.Tensor:
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
    z = torch.matmul(X, weights) + bias
    
    # 2. Apply sigmoid function
    probabilities = 1.0 / (1.0 + torch.exp(-z))
    
    # 3. Apply threshold of 0.5
    predictions = (probabilities >= 0.5).to(torch.int32)
    
    return predictions
