import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 需要的复数旋转因子 (cos + i*sin)
    
    参数:
    dim: 每一个 head 的特征维度 (head_dim)
    end: 最大序列长度 (seq_len)
    theta: 基础频率 (Llama2 是 10000，Llama3 可能是 500000)
    """
    # 频率序列: [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 位置序列: [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # 结合位置和频率: t * freqs -> 形状 (end, dim//2)
    freqs = torch.outer(t, freqs)
    # 转为复数形式极坐标: cos(freqs) + i * sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    将预计算的频率矩阵变形成可以和输入 x 广播相乘的形状
    输入 x 的形状: (batch_size, seq_len, num_heads, head_dim)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 将形状变为 (1, seq_len, 1, head_dim) 以便在 batch_size 和 num_heads 维度上广播
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用旋转位置编码 (RoPE)
    """
    # 把最后一个维度 (head_dim) reshape 出来两两一组，方便当成复数的实部和虚部
    # xq_, xk_ 形状: (batch_size, seq_len, num_heads, head_dim // 2, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整 freqs_cis 形状并进行复数乘法，这相当于旋转角度的加法 (实数域的二维旋转矩阵相乘)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

if __name__ == "__main__":
    batch_size, seq_len, num_heads, head_dim = 2, 10, 4, 16
    xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # 预计算到 seq_len 长度的旋转因子
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    
    # 应用 RoPE
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    print("Xq rotated shape:", xq_rot.shape)
    print("Xk rotated shape:", xk_rot.shape)
