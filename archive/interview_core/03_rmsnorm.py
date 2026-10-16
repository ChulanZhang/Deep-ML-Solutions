import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    均方根层归一化 (RMSNorm) - Llama / Mistral / Gemma 标配归一化
    核心考点:
    1. 公式: x / sqrt(mean(x^2) + eps) * weight
    2. 对比 LayerNorm: 舍弃了“减去均值 (mean centering)”的操作，假设激活值均值接近 0
    3. 优势: 减少了一次全局均值计算与减法，GPU 访存减少，计算速度提升 20%~30% 且表现无损
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 缩放因子 gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算沿最后一个维度的均方根 (RMS)
        # x.pow(2).mean(-1, keepdim=True) 计算每个 token 的平方均值
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 2. 归一化并乘上可学习权重 (无 bias)
        return (x / rms) * self.weight

if __name__ == "__main__":
    x = torch.randn(2, 4, 16)
    norm = RMSNorm(dim=16)
    out = norm(x)
    print("RMSNorm Output Shape:", out.shape)  # [2, 4, 16]
