def relu(z: float) -> float:
    """
    Implements the Rectified Linear Unit (ReLU) activation function.
    
    Formula: relu(z) = max(0, z)
    
    Args:
        z: Float input value
    
    Returns:
        The input if it's greater than 0, otherwise 0.
    """
    return max(0.0, float(z))
