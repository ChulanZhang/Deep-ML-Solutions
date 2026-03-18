import numpy as np

def qlora_forward(x: list[list[float]], quantized_W: list[list[int]], scale: float, zero_point: float, A: list[list[float]], B: list[list[float]], alpha: float = 1.0) -> list[list[float]]:
    """
    QLoRA forward pass with 4-bit quantized frozen weights.
    """
    X_arr = np.array(x, dtype=float)
    QW_arr = np.array(quantized_W, dtype=float)
    A_arr = np.array(A, dtype=float)
    B_arr = np.array(B, dtype=float)
    
    # Dequantize W: W = QW * scale + zero_point
    W_dq = QW_arr * scale + zero_point
    
    rank = A_arr.shape[0]
    scaling = alpha / rank
    
    frozen_path = np.dot(X_arr, W_dq)
    lora_path = np.dot(np.dot(X_arr, B_arr), A_arr) * scaling
    
    out = frozen_path + lora_path
    return np.round(out, 4).tolist()
