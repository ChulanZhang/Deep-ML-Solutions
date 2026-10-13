import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
RMSNorm = SourceFileLoader("RMSNorm", os.path.join(dir_path, "1-RMSNorm.py")).load_module().RMSNorm
precompute_freqs_cis = SourceFileLoader("RoPE", os.path.join(dir_path, "2-RoPE.py")).load_module().precompute_freqs_cis
TransformerBlock = SourceFileLoader("TransformerBlock", os.path.join(dir_path, "5-TransformerBlock.py")).load_module().TransformerBlock

class LlamaModel(nn.Module):
    """
    完整的类似 Llama 的大语言模型架构，从头到尾拼装：
    1. Token Embedding
    2. 多个 Transformer Block (含 GQA, SwiGLU, RMSNorm)
    3. 最终的 Output RMSNorm
    4. 预测下一个词的 Linear Logits 分类器
    """
    def __init__(
        self, 
        vocab_size: int = 32000, 
        dim: int = 128,          # 模型特征维度
        hidden_dim: int = 512,   # FFN 的隐藏维度
        n_layers: int = 4,       # Transformer 层数
        n_heads: int = 8,        # 注意力头数
        n_kv_heads: int = 2,     # KV 头数 (用于 GQA)
        max_seq_len: int = 1024  # 最大序列长度
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # 构建多个堆叠的 Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads) 
            for _ in range(n_layers)
        ])
        
        # 最后的全局归一化层
        self.norm = RMSNorm(dim)
        
        # 输出分类器 (Logits)
        # 根据 Llama 的标准实现，通常 bias=False
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 为了能够在每一层共享相同的旋转位置嵌入，我们将它预计算并保存为 buffer
        freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        # 暂时不用 buffer，等 forward 的时候根据序列长度截取
        self.freqs_cis = freqs_cis

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        前向传播
        tokens: 形状为 (batch_size, seq_len) 的 token id 整数矩阵
        start_pos: 当前生成的起始位置 (用于增量生成/KV Cache时，此处演示简化版)
        """
        batch_size, seq_len = tokens.shape
        
        # 1. 词嵌入 (batch_size, seq_len, dim)
        h = self.tok_embeddings(tokens)
        
        # 2. 截取对应序列长度的 RoPE (这里假设不分批推理 KV Cache，单纯跑前向)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(h.device)
        
        # 构建因果掩码 (Causal Mask)，确保自回归模型只看到过去的 token
        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        # 3. 依次通过每一个 Transformer 层
        for layer in self.layers:
            # 注意: 真正的实现中要将 freqs_cis 传给 GQA，目前我们在5-TransformerBlock中暂时占位
            h = layer(h, freqs_cis, mask)
            
        # 4. 最后的归一化
        h = self.norm(h)
        
        # 5. 生成预测每个词的对数概率 (Logits)
        # (batch_size, seq_len, vocab_size)
        logits = self.output(h)
        
        return logits

if __name__ == "__main__":
    # 测试完整的 Llama 模型前向传播
    model = LlamaModel(
        vocab_size=1000, 
        dim=128, 
        hidden_dim=512, 
        n_layers=2, 
        n_heads=8, 
        n_kv_heads=2
    )
    
    # 假装有一个 batch_size=2，长度为 10 的 token 序列
    dummy_tokens = torch.randint(0, 1000, (2, 10))
    
    # 前向传播得到 logits
    logits = model(dummy_tokens)
    
    print("Tokens shape:", dummy_tokens.shape)
    print("Output logits shape:", logits.shape)
    # Output should be (2, 10, 1000)
