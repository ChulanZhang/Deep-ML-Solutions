import numpy as np

def calculate_contrast(img):
    img_arr = np.array(img)
    if img_arr.size == 0:
        return 0.0
    return float(np.max(img_arr) - np.min(img_arr))
