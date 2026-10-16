"""Deep-ML #85: Positional Encoding Calculator.
https://www.deep-ml.com/problems/85
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def pos_encoding(position: int, d_model: int):
    if position <= 0 or d_model <= 0:
        return -1
    positions = torch.arange(position, dtype=torch.float64)[:, None]
    freq = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
    angle = positions * freq
    out = torch.empty(position, d_model, dtype=torch.float64)
    out[:, 0::2] = angle.sin()
    out[:, 1::2] = angle[:, :d_model // 2].cos()
    return out.to(torch.float16)
