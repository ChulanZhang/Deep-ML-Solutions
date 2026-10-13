import torch

def per_channel_quantization(W: torch.Tensor) -> tuple:
    W_arr = torch.tensor(W, dtype=float)
    
    max_vals = torch.amax(torch.abs(W_arr), dim=-1, keepdim=True)
    scales = 127.0 / (max_vals + 1e-12)
    
    quantized = torch.clamp(torch.round(W_arr * scales), -128, 127).astype(np.int8)
    return quantized, torch.round(scales, 4)
