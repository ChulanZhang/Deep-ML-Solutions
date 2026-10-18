import torch
import torch.nn as nn
import math

class AttentionWithKVCache(nn.Module):
    """
    带 KV Cache 的自回归注意力 - 大模型推理 (Inference) 必考题
    核心考点:
    1. Prefill 阶段: 输入全序列 (S > 1)，计算并缓存全量 K, V
    2. Decode 阶段: 每次只输入 1 个新 token (S = 1)，只计算当前 token 的 Q, K, V，
       然后将新 K, V 与历史缓存拼接: torch.cat([k_cache, k_new], dim=2)
    3. 复杂度下降: 每步生成计算量从 O(S^2) 降到 O(S)，大幅提升推理吞吐
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: tuple = None) -> tuple:
        B, S, D = x.shape  # Decode 时 S 通常为 1
        
        # 1. 计算当前输入的 Q, K, V
        Q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 拼接历史 KV Cache
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            K = torch.cat([k_prev, K], dim=2)  # 沿序列长度 seq_len 拼接
            V = torch.cat([v_prev, V], dim=2)
        
        new_kv_cache = (K, V)

        # 3. 计算注意力: Q 是 [B, H, 1, d_k]，K 是 [B, H, S_total, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # [B, H, 1, d_k]

        # 4. 恢复并输出
        out = context.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), new_kv_cache

if __name__ == "__main__":
    attn = AttentionWithKVCache(d_model=16, n_heads=4)
    # 模拟 Prefill: 提示词长度 5
    prompt = torch.randn(1, 5, 16)
    out, cache = attn(prompt, kv_cache=None)
    print("Prefill Cache K shape:", cache[0].shape)  # [1, 4, 5, 4]

    # 模拟 Decode: 生成第 1 个新 token (序列长度 1)
    new_token = torch.randn(1, 1, 16)
    out_step1, cache = attn(new_token, kv_cache=cache)
    print("Decode Step 1 Cache K shape:", cache[0].shape)  # [1, 4, 6, 4]
