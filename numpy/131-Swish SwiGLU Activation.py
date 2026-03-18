import numpy as np

def swiglu(x: np.ndarray) -> np.ndarray:
    """
    Compute SwiGLU 
    x: (B, 2d) -> Split to (B, d) and (B, d)
    """
    d = x.shape[1] // 2
    gate = x[:, :d]
    linear = x[:, d:]
    
    # Swish / SiLU
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    swish = gate * sigmoid
    
    return swish * linear
