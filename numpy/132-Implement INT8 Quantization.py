import numpy as np

def int8_quantize(x: list[float]) -> dict:
    """
    Perform symmetric INT8 quantization on a floating-point array.
    Range is [-127, 127]
    """
    if not x:
        return {'quantized': [], 'scale': 0.0, 'dequantized': []}
        
    x_arr = np.array(x, dtype=float)
    max_abs = np.max(np.abs(x_arr))
    
    if max_abs == 0:
        return {'quantized': [0]*len(x), 'scale': 0.0, 'dequantized': [0.0]*len(x)}
        
    scale = max_abs / 127.0
    
    q = np.round(x_arr / scale)
    q = np.clip(q, -127, 127).astype(int)
    
    dq = q * scale
    
    return {
        'quantized': q.tolist(),
        'scale': round(float(scale), 6),
        'dequantized': np.round(dq, 4).tolist()
    }
