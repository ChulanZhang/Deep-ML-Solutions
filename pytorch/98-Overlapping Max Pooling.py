import torch
import math

def overlapping_max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2) -> torch.Tensor:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).
    """
    N, C, H, W = x.shape
    H_out = math.ceil((H - kernel_size) / stride) + 1
    W_out = math.ceil((W - kernel_size) / stride) + 1
    
    out = torch.zeros((N, C, H_out, W_out), dtype=float)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = min(h_start + kernel_size, H)
                    w_start = j * stride
                    w_end = min(w_start + kernel_size, W)
                    
                    out[n, c, i, j] = torch.amax(x[n, c, h_start:h_end, w_start:w_end])
                    
    return out
