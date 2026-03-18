import numpy as np

def lora_forward(x: list[list[float]], W: list[list[float]], A: list[list[float]], B: list[list[float]], alpha: float = 1.0) -> list[list[float]]:
    """
    Compute the LoRA forward pass.
    """
    X_arr = np.array(x, dtype=float)
    W_arr = np.array(W, dtype=float)
    A_arr = np.array(A, dtype=float)
    B_arr = np.array(B, dtype=float)
    
    rank = A_arr.shape[0]
    scaling = alpha / rank
    
    frozen_path = np.dot(X_arr, W_arr)
    lora_path = np.dot(np.dot(X_arr, B_arr), A_arr) * scaling
    
    out = frozen_path + lora_path
    return np.round(out, 4).tolist()
