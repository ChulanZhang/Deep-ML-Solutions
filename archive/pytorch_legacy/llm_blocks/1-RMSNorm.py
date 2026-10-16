import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    相比于传统的 LayerNorm，RMSNorm 省去了计算均值的步骤，计算量更小，
    在 Llama、Qwen 等现代大模型中被广泛使用。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 学习参数 gamma，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 计算特征维度的均方根 (RMS)
        # x.pow(2).mean(-1, keepdim=True) 计算最后一个维度的平方均值
        # rsqrt = 1 / sqrt(...)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 为了保证计算精度，通常将 x 转换成 float32 计算 norm，之后再转回原来的数据类型
        output = self._norm(x.float()).type_as(x)
        # 乘以可学习的缩放参数
        return output * self.weight

if __name__ == "__main__":
    # 测试 RMSNorm
    x = torch.randn(2, 4, 8)  # (batch_size, seq_len, dim)
    norm = RMSNorm(8)
    out = norm(x)
    print("RMSNorm output shape:", out.shape)
