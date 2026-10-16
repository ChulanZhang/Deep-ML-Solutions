"""Centered PCA via thin SVD. Related: #19, but check that problem's output contract.
Singular-vector signs are not unique; compare subspaces or reconstruction, not signs.
"""
import torch

def pca(X: torch.Tensor, n_components: int):
    if not 1 <= n_components <= min(X.shape):
        raise ValueError('n_components exceeds the thin-SVD dimension.')
    centered = X - X.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:n_components].T
