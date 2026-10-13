import torch

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    is_scalar = np.isscalar(parameter) or (isinstance(parameter, torch.Tensor) and parameter.ndim == 0)
    
    p = torch.tensor(parameter, dtype=float)
    g = torch.tensor(grad, dtype=float)
    m_val = torch.tensor(m, dtype=float)
    u_val = torch.tensor(u, dtype=float)
    
    m_new = beta1 * m_val + (1 - beta1) * g
    u_new = torch.maximum(beta2 * u_val, torch.abs(g))
    
    m_hat = m_new / (1 - beta1 ** t)
    p_new = p - (learning_rate / (u_new + epsilon)) * m_hat
    
    if is_scalar:
        return float(p_new), float(m_new), float(u_new)
    return p_new, m_new, u_new
