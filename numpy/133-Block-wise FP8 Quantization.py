import numpy as np

def fp8_block_quantize(tensor: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    Max val for E4M3 is 448.
    """
    flat = tensor.flatten()
    n = len(flat)
    
    # Pad to multiple of block_size if necessary
    num_blocks = int(np.ceil(n / block_size))
    padded_len = num_blocks * block_size
    padded = np.pad(flat, (0, padded_len - n), mode='constant')
    
    blocks = padded.reshape(num_blocks, block_size)
    scales = np.max(np.abs(blocks), axis=1) / 448.0
    
    scales[scales == 0] = 1.0 # avoid div by zero
    
    q_blocks = blocks / scales[:, None]
    q_blocks = np.clip(np.round(q_blocks), -448, 448) # Simulate E4M3 range limits
    
    # In reality FP8 precision has specific mantissa/exponent rounding,
    # but the problem specifies scaling and clipping to [-448, 448].
    # We will just return the scaled rounded values.
    
    q_flat = q_blocks.flatten()[:n]
    q_tensor = q_flat.reshape(tensor.shape)
    
    return q_tensor, scales
