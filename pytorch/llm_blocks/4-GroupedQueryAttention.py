import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力机制 (Grouped Query Attention, GQA)
    - 如果 n_kv_heads == n_heads，这就是标准的多头注意力 (MHA)。
    - 如果 n_kv_heads == 1，这就是多查询注意力 (MQA)。
    - Llama 2 (大尺寸模型)、Llama 3、Qwen 等现代模型都在使用 GQA 来减少推理时的 KV Cache 内存。
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        # 为 Query, Key, Value 定义线性层
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        # 1. 经过线性投影映射到 Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 2. Reshape 成 (batch, seq_len, n_heads, head_dim) 形式
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 这里应当应用 RoPE，如果存在的话。
        # 此处假设外面有 apply_rotary_emb 函数
        # if freqs_cis is not None:
        #     xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 3. 将 K 和 V 扩展 (Broadcast) 到和 Q 相同的头数
        # 也就是 GQA 的核心：复制 KV 供多个 Q 共享
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(batch_size, seq_len, self.n_kv_heads, self.n_rep, self.head_dim)
            xk = xk.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            
            xv = xv[:, :, :, None, :].expand(batch_size, seq_len, self.n_kv_heads, self.n_rep, self.head_dim)
            xv = xv.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # 为了使用 torch.matmul，我们需要把维度转换为 (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 4. 计算注意力分数 Attention = Softmax(Q * K^T / sqrt(d)) * V
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Mask 通常是将不需要关注的地方加上极小的负数
            scores = scores + mask

        probs = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)  # (batch, n_heads, seq_len, head_dim)

        # 5. 把维度还原回并合并所有的头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 6. 最后通过输出线性层
        return self.wo(output)

if __name__ == "__main__":
    x = torch.randn(2, 10, 128)
    # Llama3风格 GQA：128维，8头查询，2头KV
    gqa = GroupedQueryAttention(dim=128, n_heads=8, n_kv_heads=2)
    out = gqa(x)
    print("GQA output shape:", out.shape)
