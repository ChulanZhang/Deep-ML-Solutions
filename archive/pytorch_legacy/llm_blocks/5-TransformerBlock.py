import torch
import torch.nn as nn

# 为了演示，我们将依赖项导入进来 (在同级目录需配合相对导入或加入路径，这里假设整合到一个完整项目)
# 从先前的模块中假想导入：
# from .rms_norm import RMSNorm
# from .swiglu_ffn import SwiGLUFFN
# from .gqa import GroupedQueryAttention

# 为了使得这个脚本直接可运行，我们先采用假的占位，实际项目中直接 import
from importlib.machinery import SourceFileLoader
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
RMSNorm = SourceFileLoader("RMSNorm", os.path.join(dir_path, "1-RMSNorm.py")).load_module().RMSNorm
SwiGLUFFN = SourceFileLoader("SwiGLUFFN", os.path.join(dir_path, "3-SwiGLU_FFN.py")).load_module().SwiGLUFFN
GroupedQueryAttention = SourceFileLoader("GroupedQueryAttention", os.path.join(dir_path, "4-GroupedQueryAttention.py")).load_module().GroupedQueryAttention

class TransformerBlock(nn.Module):
    """
    典型的预归一化 (Pre-Norm) Transformer Block。
    现代 LLM (如 Llama) 大多采用 Pre-Norm 结构，即：
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
    """
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        # 1. 注意力层前的归一化
        self.attention_norm = RMSNorm(dim)
        # 2. 注意力机制
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        
        # 3. 前馈层前的归一化
        self.ffn_norm = RMSNorm(dim)
        # 4. SwiGLU 前馈网络
        self.ffn = SwiGLUFFN(dim, hidden_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 这里的输入 x 就是残差连接 (Residual Connection) 的起点
        
        # 第一个残差分支: 注意力机制
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        
        # 第二个残差分支: 前馈神经网络
        out = h + self.ffn(self.ffn_norm(h))
        
        return out

if __name__ == "__main__":
    x = torch.randn(2, 10, 128)
    block = TransformerBlock(dim=128, hidden_dim=512, n_heads=8, n_kv_heads=2)
    out = block(x)
    print("Transformer Block output shape:", out.shape)
