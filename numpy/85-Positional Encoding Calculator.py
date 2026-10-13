import numpy as np

def pos_encoding(position: int, d_model: int):
    """
    Calculate the positional encoding vector for a given position using the sine and cosine formulas.
    """
    pos_encoding = np.zeros(d_model, dtype=float)
    
    for i in range(d_model // 2):
        denom = 10000.0 ** (2.0 * i / d_model)
        pos_encoding[2 * i] = np.sin(position / denom)
        pos_encoding[2 * i + 1] = np.cos(position / denom)
        
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding
