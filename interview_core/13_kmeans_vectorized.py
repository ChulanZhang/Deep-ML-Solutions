"""K-means with vectorized distance and index_add_ centroid updates.
Empty clusters keep their old center. Returned labels match returned centers.
Related: #17; this module has a different initialization/signature.
"""
import torch

def kmeans(X: torch.Tensor, K: int, max_iters: int = 50, tol: float = 1e-4):
    count, width = X.shape
    if not 1 <= K <= count or max_iters < 1 or tol < 0:
        raise ValueError('Invalid K, max_iters or tolerance.')
    centers = X[torch.randperm(count, device=X.device)[:K]].clone()
    for _ in range(max_iters):
        labels = (X[:, None] - centers[None]).square().sum(-1).argmin(-1)
        sums = torch.zeros_like(centers).index_add_(0, labels, X)
        counts = torch.bincount(labels, minlength=K).to(X.dtype)[:, None]
        updated = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
        converged = torch.linalg.vector_norm(updated - centers) < tol
        centers = updated
        if converged:
            break
    labels = (X[:, None] - centers[None]).square().sum(-1).argmin(-1)
    return centers, labels
