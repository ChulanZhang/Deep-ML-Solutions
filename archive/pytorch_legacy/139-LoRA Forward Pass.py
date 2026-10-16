import torch

def lora_forward(x: torch.Tensor, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Compute the LoRA forward pass.
    """
    X_arr = torch.tensor(x, dtype=float)
    W_arr = torch.tensor(W, dtype=float)
    A_arr = torch.tensor(A, dtype=float)
    B_arr = torch.tensor(B, dtype=float)
    
    rank = A_arr.shape[0]
    scaling = alpha / rank
    
    frozen_path = torch.matmul(X_arr, W_arr)
    lora_path = torch.matmul(torch.matmul(X_arr, B_arr), A_arr) * scaling
    
    out = frozen_path + lora_path
    return torch.round(out, 4).tolist()
