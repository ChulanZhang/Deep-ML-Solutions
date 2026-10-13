import torch

def qlora_forward(x: torch.Tensor, quantized_W: torch.Tensor, scale: float, zero_point: float, A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    QLoRA forward pass with 4-bit quantized frozen weights.
    """
    X_arr = torch.tensor(x, dtype=float)
    QW_arr = torch.tensor(quantized_W, dtype=float)
    A_arr = torch.tensor(A, dtype=float)
    B_arr = torch.tensor(B, dtype=float)
    
    # Dequantize W: W = QW * scale + zero_point
    W_dq = QW_arr * scale + zero_point
    
    rank = A_arr.shape[0]
    scaling = alpha / rank
    
    frozen_path = torch.matmul(X_arr, W_dq)
    lora_path = torch.matmul(torch.matmul(X_arr, B_arr), A_arr) * scaling
    
    out = frozen_path + lora_path
    return torch.round(out, 4).tolist()
