import torch

def pca(X: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    主成分分析 (PCA) - 通过 SVD 分解实现
    核心考点:
    1. 中心化 (Zero-centering): X_centered = X - mean(X, dim=0)
    2. SVD 奇异值分解: X_centered = U @ S @ V.T
       右奇异向量 V 的列向量即为协方差矩阵的特征向量 (主成分方向)
    3. 为什么优先使用 SVD 而非协方差矩阵特征分解？
       直接算 X^T @ X 会平方条件数 (Condition Number)，造成数值精度损失，SVD 更数值稳定
    4. 投影降维: X_proj = X_centered @ V[:, :n_components]
    """
    # 1. 中心化
    mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - mean

    # 2. SVD 分解
    # U: [N, D], S: [D], V: [D, D] (注意 torch.linalg.svd 返回的是 Vh 即 V.T)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh.T  # 右奇异向量矩阵

    # 3. 选取前 n_components 个主成分进行正交投影
    top_components = V[:, :n_components]
    X_projected = torch.matmul(X_centered, top_components)

    return X_projected

if __name__ == "__main__":
    X = torch.randn(10, 5)
    X_2d = pca(X, n_components=2)
    print("PCA Output Shape:", X_2d.shape)  # [10, 2]
