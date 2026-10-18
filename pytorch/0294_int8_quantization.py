"""Deep-ML #294: Implement INT8 Quantization.
https://www.deep-ml.com/problems/294
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def int8_quantize(x: list[float]) -> dict:
    # The PyTorch task uses float32; half-way rounding is dtype-sensitive.
    values = torch.as_tensor(x, dtype=torch.float32)
    maximum = values.abs().max().item() if values.numel() else 0.0
    scale = maximum / 127 if maximum else 1.0
    q = torch.round(values / scale).clamp(-127, 127).to(torch.int8)
    return {'quantized': q.tolist(), 'scale': round(scale, 6),
            'dequantized': [round(v, 4) for v in (q.float() * scale).tolist()]}
