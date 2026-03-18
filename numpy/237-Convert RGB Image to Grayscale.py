import numpy as np

def rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    img = np.array(img, dtype=float)
    # Using standard ITU-R 601-2 luma transform
    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return np.round(gray, 4)
