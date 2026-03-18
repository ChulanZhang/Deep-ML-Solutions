import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    """
    # Check if array or scalar for type consistency calculation
    is_scalar = not isinstance(parameter, np.ndarray) and not isinstance(parameter, list)
    
    v = np.array(velocity, dtype=float)
    p = np.array(parameter, dtype=float)
    g = np.array(grad, dtype=float)
    
    v_new = momentum * v + learning_rate * g
    p_new = p - v_new
    
    if is_scalar:
        return float(p_new), float(v_new)
    
    return p_new, v_new
