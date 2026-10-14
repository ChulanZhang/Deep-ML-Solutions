import torch

def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0):
    """
    预计算旋转角度的 cos 和 sin
    频率: theta_i = base^(-2i / dim)
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [max_seq_len, dim // 2]
    return torch.cos(angles), torch.sin(angles)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    旋转位置编码 (RoPE) - 面试标准模版
    核心考点:
    1. 原理: 将特征两两分组并在 2D 复平面上旋转，使得点积 <R_m*q, R_n*k> 只与相对位置 (m-n) 相关
    2. 旋转矩阵应用公式:
       x_rot = [-x2, x1]
       output = x * cos + x_rot * sin
    """
    # x 形状: [Batch, Seq_len, n_heads, head_dim]
    B, S, H, D = x.shape
    
    # 截取当前序列长度对应的 cos 和 sin，并广播到 [1, S, 1, D // 2]
    cos = cos[:S, :].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]
    sin = sin[:S, :].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]

    # 将最后两两一组切分为前半段与后半段
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]

    # 旋转变换: [x1, x2] 乘以旋转矩阵 -> [x1*cos - x2*sin, x2*cos + x1*sin]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    return torch.cat([out1, out2], dim=-1)

if __name__ == "__main__":
    B, S, H, D = 2, 8, 4, 16
    q = torch.randn(B, S, H, D)
    cos, sin = precompute_freqs_cis(dim=D, max_seq_len=S)
    q_rot = apply_rotary_emb(q, cos, sin)
    print("RoPE Output Shape:", q_rot.shape)  # [2, 8, 4, 16]
