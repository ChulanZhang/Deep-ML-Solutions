import torch

def kmeans(X: torch.Tensor, K: int, max_iters: int = 50, tol: float = 1e-4):
    """
    K-Means 聚类 (纯向量化实现) - 机器学习面试常考无监督算法
    核心考点:
    1. 距离计算的广播技巧: X[:, None, :] - centers[None, :, :]
       得到形状 [N, K, D]，平方求和即可一次性算出所有样本到所有中心的欧式距离
    2. 簇分配: torch.argmin(dists, dim=-1)
    3. 中心更新: 取各簇内样本均值
    """
    N, D = X.shape
    # 1. 随机选取 K 个样本作为初始中心
    perm = torch.randperm(N)
    centers = X[perm[:K]].clone()

    for _ in range(max_iters):
        # 2. 广播计算距离平方: [N, 1, D] - [1, K, D] -> [N, K]
        diff = X.unsqueeze(1) - centers.unsqueeze(0)
        dists = torch.sum(diff ** 2, dim=-1)

        # 3. 分配到最近的中心
        labels = torch.argmin(dists, dim=-1)

        # 4. 更新中心
        new_centers = torch.stack([
            X[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centers[k]
            for k in range(K)
        ])

        # 5. 检查是否收敛
        if torch.norm(new_centers - centers) < tol:
            break
        centers = new_centers

    return centers, labels

if __name__ == "__main__":
    X = torch.cat([torch.randn(20, 2) + 5, torch.randn(20, 2) - 5], dim=0)
    centers, labels = kmeans(X, K=2)
    print("K-Means Centers:\n", centers)
