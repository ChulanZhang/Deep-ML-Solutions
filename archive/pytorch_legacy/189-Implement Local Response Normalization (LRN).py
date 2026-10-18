import torch

def local_response_normalization(x: torch.Tensor, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> torch.Tensor:
    x_arr = torch.tensor(x, dtype=float)
    res = torch.zeros_like(x_arr)
    C = x_arr.shape[1]
    
    for c in range(C):
        c_start = max(0, c - n // 2)
        c_end = min(C, c + n // 2 + 1)
        sq_sum = torch.sum(x_arr[:, c_start:c_end, :, :] ** 2, dim=1)
        res[:, c, :, :] = x_arr[:, c, :, :] / ((k + alpha * sq_sum) ** beta)
        
    return res
