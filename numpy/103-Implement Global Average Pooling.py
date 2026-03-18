import numpy as np

def global_average_pooling(x):
    """
    Computes global average pooling for a 3D numpy array (H, W, C).
    Output shape: (channels,)
    """
    x = np.array(x, dtype=float)
    return np.mean(x, axis=(0, 1))
