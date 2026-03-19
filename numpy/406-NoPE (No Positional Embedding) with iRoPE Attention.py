import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    """
    seq_len, d = x.shape
    half_d = d // 2
    
    theta = base ** (-2 * np.arange(half_d) / d)
    m_theta = positions[:, None] * theta[None, :]
    
    cos_val = np.cos(m_theta)
    sin_val = np.sin(m_theta)
    
    out = np.zeros_like(x)
    
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    
    out[:, 0::2] = x_even * cos_val - x_odd * sin_val
    out[:, 1::2] = x_even * sin_val + x_odd * cos_val
    
    return out
