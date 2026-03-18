import numpy as np

def conv3d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    image_arr = np.array(image, dtype=float)
    kernel_arr = np.array(kernel, dtype=float)
    
    if padding > 0:
        image_arr = np.pad(image_arr, padding, mode='constant', constant_values=0)
        
    D, H, W = image_arr.shape
    kD, kH, kW = kernel_arr.shape
    
    out_D = (D - kD) // stride + 1
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    out = np.zeros((out_D, out_H, out_W))
    for d in range(out_D):
        for h in range(out_H):
            for w in range(out_W):
                d_s = d * stride
                h_s = h * stride
                w_s = w * stride
                
                patch = image_arr[d_s:d_s+kD, h_s:h_s+kH, w_s:w_s+kW]
                out[d, h, w] = np.sum(patch * kernel_arr)
                
    return out
