import numpy as np

def video_memory_patchify(video: np.ndarray, t_patch: int, h_patch: int, w_patch: int) -> np.ndarray:
    video = np.array(video, dtype=float)
    T, C, H, W = video.shape
    
    out_t, out_h, out_w = T // t_patch, H // h_patch, W // w_patch
    patches = []
    
    for i in range(out_t):
        for j in range(out_h):
            for k in range(out_w):
                patch = video[i*t_patch:(i+1)*t_patch, :, j*h_patch:(j+1)*h_patch, k*w_patch:(k+1)*w_patch]
                patches.append(patch.flatten())
                
    return np.round(np.array(patches), 4)
