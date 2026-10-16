import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    低秩自适应层 (LoRA) - 大模型微调 (PEFT) 必考题
    核心考点:
    1. 公式: y = W_0 * x + (alpha / r) * (B @ A @ x)
    2. 参数初始化技巧:
       - A 使用高斯初始化 (或 Kaiming 均匀分布)
       - B 必须初始化为全 0 (确保训练初始时 delta_W = 0，完全保持原模型行为)
    3. 缩放因子: alpha / r，当尝试不同的秩 r 时无需大幅调节学习率
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        # 冻结的原模型权重 W_0
        self.pretrained = nn.Linear(in_features, out_features, bias=False)
        self.pretrained.weight.requires_grad = False  # 冻结主权重

        self.r = r
        self.scaling = lora_alpha / r

        # LoRA 的低秩矩阵: A 降维到 r，B 升维到 out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # 关键: 初始化全 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支: 原始冻结权重输出
        orig_out = self.pretrained(x)
        
        # 旁路低秩分支: x @ A.T @ B.T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return orig_out + lora_out

if __name__ == "__main__":
    x = torch.randn(2, 4, 16)
    lora = LoRALinear(in_features=16, out_features=32, r=4)
    out = lora(x)
    print("LoRA Output Shape:", out.shape)  # [2, 4, 32]
