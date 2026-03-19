import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Quantize a tensor to MXFP4 format using microscaling.
    Max limit for simulated 4-bit (e.g., E2M1 or similar) is typically ~ 6.0 or 7.0 depending on the spec, 
    but we'll assume a symmetric [-7, 7] or similar standard. 
    Let's scale to max_abs / 7.0 for 4-bit representation limit.
    """
    if not x:
        return {'quantized': [], 'scales': [], 'dequantized': []}
        
    n = len(x)
    arr = np.array(x, dtype=float)
    
    # Pad
    num_blocks = int(np.ceil(n / block_size))
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        arr = np.pad(arr, (0, pad_len))
        
    blocks = arr.reshape((num_blocks, block_size))
    scales = np.max(np.abs(blocks), axis=1) / 6.0 # standard E2M1 max is ~6.0
    scales[scales == 0] = 1.0
    
    q_blocks = np.clip(np.round(blocks / scales[:, None]), -6, 6)
    dq_blocks = q_blocks * scales[:, None]
    
    q_flat = q_blocks.flatten()[:n].astype(int).tolist()
    dq_flat = dq_blocks.flatten()[:n].tolist()
    scales_list = scales.tolist()
    
    return {
        'quantized': q_flat,
        'scales': scales_list,
        'dequantized': [round(v, 4) for v in dq_flat]
    }
