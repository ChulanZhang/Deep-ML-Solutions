def leaky_relu(z: float, alpha: float = 0.01) -> float:
    """
    Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    
    Formula: leaky_relu(z) = z if z > 0 else alpha * z
    
    Args:
        z: Float input value
        alpha: Slope for negative inputs (default 0.01)
        
    Returns:
        The Leaky ReLU of z.
    """
    if z > 0:
        return float(z)
    else:
        return float(alpha * z)
