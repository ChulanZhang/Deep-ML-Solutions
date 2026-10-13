import torch

def ffn(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, dropout_p: float = 0.1, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    x_arr = torch.tensor(x, dtype=float)
    W1 = torch.tensor(W1, dtype=float)
    b1 = torch.tensor(b1, dtype=float)
    W2 = torch.tensor(W2, dtype=float)
    b2 = torch.tensor(b2, dtype=float)
    
    # Linear 1
    hidden = torch.matmul(x_arr, W1) + b1
    # ReLU
    hidden = torch.maximum(0, hidden)
    # Dropout
    if dropout_p > 0:
        mask = (torch.rand(*hidden.shape) >= dropout_p).to(torch.float32)
        hidden = hidden * mask / (1.0 - dropout_p)
        
    # Linear 2
    out = torch.matmul(hidden, W2) + b2
    
    # Residual
    res = out + x_arr
    return res.tolist()
