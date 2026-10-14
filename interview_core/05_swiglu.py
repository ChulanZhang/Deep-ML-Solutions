import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络 (FFN) - Llama 系列标配 MLP
    核心考点:
    1. 公式: Output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    2. 门控机制 (GLU): 通过 Gate 分支动态过滤 Up 分支的信息流
    3. 中间维度隐藏层通常取 8/3 * d_model 并对齐到 256 的倍数 (兼顾参数量与标准 4*d_model 相當)
    """
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        # 默认取 8/3 * d_model 作为隐藏层维度
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * d_model) / 3)
            
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)  # 门控分支
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)    # 信息分支
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)  # 输出降维投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w_gate(x)) * w_up(x) -> 逐元素相乘门控滤波
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

if __name__ == "__main__":
    x = torch.randn(2, 4, 16)
    ffn = SwiGLU(d_model=16, hidden_dim=32)
    out = ffn(x)
    print("SwiGLU Output Shape:", out.shape)  # [2, 4, 16]
