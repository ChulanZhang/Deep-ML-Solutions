import torch
import torch.nn.functional as F

def infonce_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    对比学习 InfoNCE 损失 (CLIP / SimCLR 经典实现)
    核心考点:
    1. 特征首先进行 L2 归一化，使得矩阵点积直接等价于余弦相似度
    2. 相似度矩阵: logits = (a @ b.T) / tau，尺寸为 [N, N]
    3. 巧妙的向量化技巧: 对角线位置 (i == j) 即为正样本对，其余非对角线为负样本
       因此直接计算 logits 与标签 torch.arange(N) 的交叉熵损失即可！
    """
    # 1. L2 归一化
    a = F.normalize(features_a, dim=-1)
    b = F.normalize(features_b, dim=-1)

    # 2. 余弦相似度矩阵与温度系数缩放
    # [N, D] @ [D, N] -> [N, N]
    logits = torch.matmul(a, b.T) / temperature

    # 3. 构造正样本标签: [0, 1, 2, ..., N-1]
    batch_size = a.shape[0]
    labels = torch.arange(batch_size, device=a.device)

    # 4. 双向交叉熵损失 (A 对 B，B 对 A 对称优化)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2.0

if __name__ == "__main__":
    feat_img = torch.randn(4, 128)
    feat_txt = torch.randn(4, 128)
    loss = infonce_loss(feat_img, feat_txt)
    print("InfoNCE Loss:", loss.item())
