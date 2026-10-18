import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (GQA) - 现代大模型 (Llama 3 / Qwen) 标配
    核心考点:
    1. Q 头数 (n_q_heads) 多于 KV 头数 (n_kv_heads)，节省 KV Cache 显存
    2. repeat_interleave 扩展 KV 头数以匹配 Q，进而做标准注意力计算
    3. 特例认知: 当 n_kv_heads = 1 为 MQA; 当 n_kv_heads = n_q_heads 为 MHA
    """
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "n_q_heads 必须能被 n_kv_heads 整除"
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.num_groups = n_q_heads // n_kv_heads  # 每个 KV 头对应几个 Q 头
        self.d_k = d_model // n_q_heads

        self.q_proj = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # 1. 投影切头: Q -> [B, n_q_heads, S, d_k], K,V -> [B, n_kv_heads, S, d_k]
        Q = self.q_proj(x).view(B, S, self.n_q_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 2. 关键步: 将 KV 沿 head 维度复制展开，与 Q 头数对齐
        # 形状变化: [B, n_kv_heads, S, d_k] -> [B, n_q_heads, S, d_k]
        K = torch.repeat_interleave(K, repeats=self.num_groups, dim=1)
        V = torch.repeat_interleave(V, repeats=self.num_groups, dim=1)

        # 3. 标准 Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # [B, n_q_heads, S, d_k]

        # 4. 拼回所有 Head 并输出投影 -> [B, S, D]
        context = context.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(context)

if __name__ == "__main__":
    B, S, D = 2, 4, 32
    # 8 个 Q 头，2 个 KV 头 (4 倍 KV Cache 压缩比)
    gqa = GroupedQueryAttention(d_model=D, n_q_heads=8, n_kv_heads=2)
    out = gqa(torch.randn(B, S, D))
    print("GQA Output Shape:", out.shape)  # 期望: [2, 4, 32]
