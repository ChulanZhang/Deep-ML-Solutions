import math

def sigmoid(z: float) -> float:
    """
    Compute the output of the sigmoid activation function given an input value z.
    
    Formula: sigmoid(z) = 1 / (1 + e^-z)
    
    Args:
        z: Float input value
        
    Returns:
        Sigmoid output rounded to 4 decimal places.
    """
    # 1. Compute the denominator: 1 + e^(-z)
    # Using math.exp for the exponential function
    denominator = 1.0 + math.exp(-z)
    
    # 2. Calculate the sigmoid value
    result = 1.0 / denominator
    
    # 3. Round to four decimal places as requested
    return round(result, 4)
