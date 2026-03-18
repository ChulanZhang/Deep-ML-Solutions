import numpy as np

def zero_padding(img: np.ndarray, pad_width: int) -> np.ndarray:
    img = np.array(img)
    if img.ndim == 2:
        return np.pad(img, pad_width, mode='constant', constant_values=0)
    elif img.ndim == 3:
        # Avoid padding channels (assume channel is last, but deep-ml vision tasks vary. If 2D image is expected, we pad H, W)
        return np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
    return img
