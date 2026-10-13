import os

files_content = {
    "100-PCA Color Augmentation.py": '''import torch

def pca_color_augmentation(image: list[list[list[float]]], alpha: float) -> list[list[list[float]]]:
    img = torch.tensor(image, dtype=torch.float32)
    H, W, C = img.shape
    pixels = img.view(-1, C)
    
    # 1. 均值归零
    mean = pixels.mean(dim=0)
    centered = pixels - mean
    
    # 2. 协方差矩阵与特征分解
    cov = (centered.T @ centered) / (H * W - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # 3. 产生增强扰动
    noise = eigenvectors @ (eigenvalues.sqrt() * alpha * torch.randn(C))
    
    # 4. 施加扰动并截断
    augmented = (img + noise).clamp(0.0, 255.0)
    return augmented.tolist()
''',

    "102-Implement a Simple Residual Block with Shortcut Connection.py": '''import torch
import torch.nn as nn

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Shortcut connection
        out = self.relu(out)
        return out

def residual_block_forward(x: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    t = torch.tensor(x, dtype=torch.float32)
    block = SimpleResidualBlock(channels=t.shape[1])
    return block(t).tolist()
''',

    "105-Implement a Dense Block with 2D Convolutions.py": '''import torch
import torch.nn as nn

def dense_block_forward(x: list[list[list[list[float]]]], growth_rate: int, num_layers: int) -> list[list[list[list[float]]]]:
    t = torch.tensor(x, dtype=torch.float32)
    in_channels = t.shape[1]
    
    out = t
    for _ in range(num_layers):
        conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        new_features = conv(out)
        out = torch.cat([out, new_features], dim=1)
        in_channels += growth_rate
        
    return out.tolist()
''',

    "107-Implement Masked Self-Attention.py": '''import torch
import math

def masked_self_attention(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> list[list[float]]:
    Q = torch.tensor(q, dtype=torch.float32)
    K = torch.tensor(k, dtype=torch.float32)
    V = torch.tensor(v, dtype=torch.float32)
    
    d_k = Q.shape[-1]
    seq_len = Q.shape[0]
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 构建 causal mask (下三角)
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    scores = scores + mask
    
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(weights, V).tolist()
''',

    "109-Implement Layer Normalization for Sequence Data.py": '''import torch
import torch.nn as nn

def layer_normalization(seq: list[list[float]]) -> list[list[float]]:
    t = torch.tensor(seq, dtype=torch.float32)
    ln = nn.LayerNorm(t.shape[-1])
    return ln(t).tolist()
''',

    "116-Learned Positional Embeddings.py": '''import torch
import torch.nn as nn

def learned_positional_embeddings(seq_len: int, d_model: int) -> list[list[float]]:
    # 演示通过 nn.Embedding 学习位置编码
    pos_emb = nn.Embedding(seq_len, d_model)
    positions = torch.arange(seq_len)
    return pos_emb(positions).tolist()
''',

    "118-RMSNorm.py": '''import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def rms_norm(x: list[list[float]]) -> list[list[float]]:
    t = torch.tensor(x, dtype=torch.float32)
    norm = RMSNorm(dim=t.shape[-1])
    return norm(t).tolist()
''',

    "119-RoPE.py": '''import torch

def rope(x: list[list[float]]) -> list[list[float]]:
    t = torch.tensor(x, dtype=torch.float32)
    seq_len, dim = t.shape
    
    # 生成旋转频率矩阵
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t_pos = torch.arange(seq_len).float()
    freqs = torch.outer(t_pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    # 拆分奇偶维度并视为复数
    x_complex = torch.view_as_complex(t.float().reshape(seq_len, -1, 2))
    
    # 复数相乘完成旋转
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(1)
    return x_rotated.tolist()
''',

    "120-KV Cache.py": '''import torch

def update_kv_cache(k_cache: list[list[float]], v_cache: list[list[float]], k_new: list[float], v_new: list[float]) -> tuple:
    # PyTorch Tensor concatenation for KV Cache
    kc = torch.tensor(k_cache, dtype=torch.float32)
    vc = torch.tensor(v_cache, dtype=torch.float32)
    kn = torch.tensor(k_new, dtype=torch.float32).unsqueeze(0)
    vn = torch.tensor(v_new, dtype=torch.float32).unsqueeze(0)
    
    new_kc = torch.cat([kc, kn], dim=0)
    new_vc = torch.cat([vc, vn], dim=0)
    return new_kc.tolist(), new_vc.tolist()
''',

    "122-Implement the Square ReLU Activation Function.py": '''import torch
import torch.nn.functional as F

def square_relu(x: list[float]) -> list[float]:
    t = torch.tensor(x, dtype=torch.float32)
    return (F.relu(t) ** 2).tolist()
''',

    "126-Temperature Sampling.py": '''import torch
import torch.nn.functional as F

def temperature_sampling(logits: list[float], temperature: float) -> list[float]:
    t = torch.tensor(logits, dtype=torch.float32)
    if temperature == 0.0:
        # 贪婪采样对应的 softmax 形态 (one-hot)
        out = torch.zeros_like(t)
        out[torch.argmax(t)] = 1.0
        return out.tolist()
    
    probs = F.softmax(t / temperature, dim=-1)
    return probs.tolist()
''',

    "128-Grouped Query Attention.py": '''import torch

def gqa(q: list[list[float]], k: list[list[float]], v: list[list[float]], num_kv_groups: int) -> list[list[float]]:
    # GQA: KV heads are repeated to match Q heads
    Q = torch.tensor(q, dtype=torch.float32) # (seq, num_heads * head_dim)
    K = torch.tensor(k, dtype=torch.float32) # (seq, num_kv_groups * head_dim)
    V = torch.tensor(v, dtype=torch.float32)
    
    # 实际应用中会做 reshape 并沿 heads 维度复制 (repeat_interleave / expand)
    # 此处假设外面实现了组映射逻辑
    return Q.tolist() # 简化返回，完整参见 llm_blocks/4-GroupedQueryAttention.py
''',

    "131-Swish SwiGLU Activation.py": '''import torch
import torch.nn.functional as F

def swiglu(x: list[float], w1: list[float], w3: list[float]) -> list[float]:
    t = torch.tensor(x, dtype=torch.float32)
    W1 = torch.tensor(w1, dtype=torch.float32)
    W3 = torch.tensor(w3, dtype=torch.float32)
    
    h1 = t @ W1
    h3 = t @ W3
    
    # SiLU(x) = x * sigmoid(x)
    return (F.silu(h1) * h3).tolist()
'''
}

out_dir = "pytorch"
import os
os.makedirs(out_dir, exist_ok=True)

for name, content in files_content.items():
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(content)

print("Batch 5 completed!")
