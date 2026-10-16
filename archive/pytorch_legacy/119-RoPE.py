import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    """
    seq_len, d = x.shape
    half_d = d // 2
    
    theta = base ** (-2 * torch.arange(half_d) / d)
    m_theta = positions[:, None] * theta[None, :]
    
    cos_val = torch.cos(m_theta)
    sin_val = torch.sin(m_theta)
    
    out = torch.zeros_like(x)
    
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    
    out[:, 0::2] = x_even * cos_val - x_odd * sin_val
    out[:, 1::2] = x_even * sin_val + x_odd * cos_val
    
    return out
