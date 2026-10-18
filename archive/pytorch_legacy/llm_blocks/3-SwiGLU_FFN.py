import torch
import torch.nn as nn

class SiLU(nn.Module):
    """
    手动实现 SiLU 激活函数 (也叫 Swish)。
    公式: SiLU(x) = x * sigmoid(x)
    说明: 用户要求尽量用基础模块搭建，不直接调用 torch.nn.functional.silu
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLUFFN(nn.Module):
    """
    基于 SwiGLU 的前馈神经网络 (Feed Forward Network)
    在 Llama 结构中，FFN 通常由三个线性层组成。
    公式: FFN(x) = (SiLU(x * W1) * (x * W3)) * W2
    这里乘法代表矩阵乘法 (Linear Layer)。
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # 用户要求用 Linear 搭建
        # W1: 用于生成 SiLU 的输入
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # W3: 用于生成 Gating (门控) 的乘数
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # W2: 将 hidden_dim 投影回原维度 dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
        # 手动实现的 SiLU
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算通过 w1 并使用 SiLU 激活的值
        # 2. 计算通过 w3 的值 (不加激活)
        # 3. 将两者逐元素相乘 (element-wise multiplication)
        # 4. 最后通过 w2 投影回原始维度
        hidden = self.silu(self.w1(x)) * self.w3(x)
        output = self.w2(hidden)
        return output

if __name__ == "__main__":
    x = torch.randn(2, 10, 32) # (batch_size, seq_len, dim)
    # 按照 Llama 习惯，hidden_dim 通常是 dim 的一定倍数 (例如 4倍或基于特定的多项式)
    ffn = SwiGLUFFN(dim=32, hidden_dim=128)
    out = ffn(x)
    print("SwiGLU FFN output shape:", out.shape)
