import torch

def ffn(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, dropout_p: float, seed: int) -> torch.Tensor:
    """
    Position-wise Feed-Forward Network (FFN) with residual connection and dropout.
    """
    torch.manual_seed(seed)
    
    # 1. Linear 1 + ReLU
    inter = torch.maximum(0, torch.matmul(x, w1) + b1)
    
    # 2. Dropout
    mask = (torch.rand(*inter.shape) >= dropout_p).to(torch.float32)
    dropped = inter * mask / (1.0 - dropout_p)
    
    # 3. Linear 2
    out = torch.matmul(dropped, w2) + b2
    
    # 4. Residual and Round
    result = torch.round(out + x, 4)
    
    return result
