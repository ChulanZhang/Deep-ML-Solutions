import numpy as np

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    image = np.array(image, dtype=float)
    h, w = image.shape
    out = np.zeros_like(image)
    
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = image[i-1:i+2, j-1:j+2]
            gx = np.sum(patch * Gx)
            gy = np.sum(patch * Gy)
            out[i, j] = np.sqrt(gx**2 + gy**2)
            
    return np.round(out, 4)
