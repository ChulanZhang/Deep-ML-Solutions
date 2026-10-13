import torch

def int8_quantize(x: torch.Tensor) -> dict:
    """
    Perform symmetric INT8 quantization on a floating-point array.
    Range is [-127, 127]
    """
    if not x:
        return {'quantized': [], 'scale': 0.0, 'dequantized': []}
        
    x_arr = torch.tensor(x, dtype=float)
    max_abs = torch.amax(torch.abs(x_arr))
    
    if max_abs == 0:
        return {'quantized': [0]*len(x), 'scale': 0.0, 'dequantized': [0.0]*len(x)}
        
    scale = max_abs / 127.0
    
    q = torch.round(x_arr / scale)
    q = torch.clamp(q, -127, 127).to(torch.int32)
    
    dq = q * scale
    
    return {
        'quantized': q.tolist(),
        'scale': round(float(scale), 6),
        'dequantized': torch.round(dq, 4).tolist()
    }
