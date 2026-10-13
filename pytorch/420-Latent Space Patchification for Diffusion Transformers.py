import torch

def latent_diff_patchify(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    latent = torch.tensor(latent, dtype=float)
    if latent.ndim == 3:  # (C, H, W)
        c, h, w = latent.shape
        out_h, out_w = h // patch_size, w // patch_size
        patches = []
        for i in range(out_h):
            for j in range(out_w):
                patch = latent[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches.append(patch.flatten())
        return torch.round(torch.tensor(patches), 4)
    return latent
