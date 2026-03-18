import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    is_scalar = np.isscalar(parameter) or (isinstance(parameter, np.ndarray) and parameter.ndim == 0)
    
    p = np.array(parameter, dtype=float)
    g = np.array(grad, dtype=float)
    m_val = np.array(m, dtype=float)
    u_val = np.array(u, dtype=float)
    
    m_new = beta1 * m_val + (1 - beta1) * g
    u_new = np.maximum(beta2 * u_val, np.abs(g))
    
    m_hat = m_new / (1 - beta1 ** t)
    p_new = p - (learning_rate / (u_new + epsilon)) * m_hat
    
    if is_scalar:
        return float(p_new), float(m_new), float(u_new)
    return p_new, m_new, u_new
