import torch

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    """
    # Check if array or scalar for type consistency calculation
    is_scalar = not isinstance(parameter, torch.Tensor) and not isinstance(parameter, list)
    
    v = torch.tensor(velocity, dtype=float)
    p = torch.tensor(parameter, dtype=float)
    g = torch.tensor(grad, dtype=float)
    
    v_new = momentum * v + learning_rate * g
    p_new = p - v_new
    
    if is_scalar:
        return float(p_new), float(v_new)
    
    return p_new, v_new
