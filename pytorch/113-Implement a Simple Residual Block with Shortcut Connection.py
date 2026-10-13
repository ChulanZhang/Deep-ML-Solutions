import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implements x + Layer2(ReLU(Layer1(x))) with final ReLU.
    """
    x_np = torch.tensor(x, dtype=float)
    w1_np = torch.tensor(w1, dtype=float)
    w2_np = torch.tensor(w2, dtype=float)
    
    # Layer 1
    z1 = w1_np @ x_np
    a1 = torch.maximum(0, z1)
    
    # Layer 2
    z2 = w2_np @ a1
    
    # Add Shortcut
    out = z2 + x_np
    
    # Final ReLU
    return torch.maximum(0, out)
