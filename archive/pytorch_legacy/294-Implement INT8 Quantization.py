import torch

def int8_quantize(W: torch.Tensor) -> tuple:
    W_arr = torch.tensor(W, dtype=float)
    max_val = torch.amax(torch.abs(W_arr))
    if max_val == 0:
        return W_arr.astype(np.int8), 1.0
    scale = 127.0 / max_val
    W_quant = torch.clamp(torch.round(W_arr * scale), -128, 127).astype(np.int8)
    return W_quant, float(torch.round(scale, 4))
