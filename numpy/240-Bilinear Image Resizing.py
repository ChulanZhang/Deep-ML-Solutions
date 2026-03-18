import numpy as np

def bilinear_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    img = np.array(img, dtype=float)
    h, w = img.shape[:2]
    out = np.zeros((new_h, new_w) + img.shape[2:])
    
    for i in range(new_h):
        for j in range(new_w):
            x = i * (h - 1) / (new_h - 1) if new_h > 1 else 0.0
            y = j * (w - 1) / (new_w - 1) if new_w > 1 else 0.0
            
            x1, y1 = int(np.floor(x)), int(np.floor(y))
            x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)
            
            dx, dy = x - x1, y - y1
            
            val = (1 - dx) * (1 - dy) * img[x1, y1] + \
                  (1 - dx) * dy * img[x1, y2] + \
                  dx * (1 - dy) * img[x2, y1] + \
                  dx * dy * img[x2, y2]
                  
            out[i, j] = val
            
    return np.round(out, 4)
