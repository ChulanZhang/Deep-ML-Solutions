import torch

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    is_scalar = np.isscalar(parameter) or (isinstance(parameter, torch.Tensor) and parameter.ndim == 0)
    
    p = torch.tensor(parameter, dtype=float)
    g = torch.tensor(grad, dtype=float)
    u_val = torch.tensor(u, dtype=float)
    v_val = torch.tensor(v, dtype=float)
    
    v_new = rho * v_val + (1 - rho) * (g ** 2)
    delta = - (torch.sqrt(u_val + epsilon) / torch.sqrt(v_new + epsilon)) * g
    p_new = p + delta
    u_new = rho * u_val + (1 - rho) * (delta ** 2)
    
    if is_scalar:
        return float(p_new), float(u_new), float(v_new)
    return p_new, u_new, v_new
