import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    经典多头注意力 (MHA) - 面试标准模版
    核心考点: 
    1. 4D 张量重塑与转置: [B, S, D] -> [B, H, S, d_k] (让 GPU 并行计算所有 head)
    2. 缩放因子 1/sqrt(d_k) 防止点积方差过大导致 softmax 梯度饱和
    3. Causal Mask 下三角掩码处理
    4. transpose 后的 .contiguous() 内存连续化
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性映射层 (面试时也可简写为一个 W_qkv 矩阵)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape

        # 1. 投影并拆分多头: [B, S, D] -> [B, S, H, d_k] -> [B, H, S, d_k]
        Q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数: [B, H, S, d_k] @ [B, H, d_k, S] -> [B, H, S, S]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 掩码 (若有因果掩码或 padding 掩码)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax 归一化并与 V 相乘 -> [B, H, S, d_k]
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 5. 拼回所有 Head 并进行最终投影 -> [B, S, D]
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(context)

if __name__ == "__main__":
    # 极简验证
    B, S, D, H = 2, 4, 16, 4
    x = torch.randn(B, S, D)
    mha = MultiHeadAttention(d_model=D, n_heads=H)
    out = mha(x)
    print("MHA Output Shape:", out.shape)  # 期望: [2, 4, 16]
