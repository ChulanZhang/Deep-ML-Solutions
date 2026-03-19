import numpy as np

def latent_diff_patchify(latent: np.ndarray, patch_size: int) -> np.ndarray:
    latent = np.array(latent, dtype=float)
    if latent.ndim == 3:  # (C, H, W)
        c, h, w = latent.shape
        out_h, out_w = h // patch_size, w // patch_size
        patches = []
        for i in range(out_h):
            for j in range(out_w):
                patch = latent[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches.append(patch.flatten())
        return np.round(np.array(patches), 4)
    return latent
