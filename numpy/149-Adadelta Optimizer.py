import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    is_scalar = np.isscalar(parameter) or (isinstance(parameter, np.ndarray) and parameter.ndim == 0)
    
    p = np.array(parameter, dtype=float)
    g = np.array(grad, dtype=float)
    u_val = np.array(u, dtype=float)
    v_val = np.array(v, dtype=float)
    
    v_new = rho * v_val + (1 - rho) * (g ** 2)
    delta = - (np.sqrt(u_val + epsilon) / np.sqrt(v_new + epsilon)) * g
    p_new = p + delta
    u_new = rho * u_val + (1 - rho) * (delta ** 2)
    
    if is_scalar:
        return float(p_new), float(u_new), float(v_new)
    return p_new, u_new, v_new
