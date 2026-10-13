def prelu(z: float, alpha: float) -> float:
    """
    Implements the Parametric Rectified Linear Unit (PReLU) activation function.
    
    Formula: prelu(z) = z if z > 0 else alpha * z
    (Note: Mathematically identical to Leaky ReLU, but alpha is treated as a 
    tunable/learnable parameter rather than a fixed hyperparameter in practice).
    
    Args:
        z: Float input value
        alpha: Tunable parameter slope for negative inputs
        
    Returns:
        The PReLU of z.
    """
    if z > 0:
        return float(z)
    else:
        return float(alpha * z)
