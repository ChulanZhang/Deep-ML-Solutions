import numpy as np

def max_pooling(image: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    image = np.array(image, dtype=float)
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            val = np.max(image[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size])
            out[i, j] = val
    return out
