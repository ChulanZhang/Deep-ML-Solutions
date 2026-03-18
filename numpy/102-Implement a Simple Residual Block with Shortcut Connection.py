import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """
    Implements x + Layer2(ReLU(Layer1(x))) with final ReLU.
    """
    x_np = np.array(x, dtype=float)
    w1_np = np.array(w1, dtype=float)
    w2_np = np.array(w2, dtype=float)
    
    # Layer 1
    z1 = w1_np @ x_np
    a1 = np.maximum(0, z1)
    
    # Layer 2
    z2 = w2_np @ a1
    
    # Add Shortcut
    out = z2 + x_np
    
    # Final ReLU
    return np.maximum(0, out)
