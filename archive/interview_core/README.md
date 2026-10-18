# 🎯 大厂算法 & LLM 面试手撕代码核心精编库 (Interview Core)

> **面试理念**：面试不是写庞大的工业级开源库，更不是做在线题库（OJ）评测。  
> 考场上，面试官要看的是你**能否在 15~20 分钟内用 20~30 行最干净的代码，把底层数学逻辑、张量维度变换（Tensor Shapes）和关键考点讲得清清楚楚**。

---

## 📚 核心题库导航清单

| 编号 | 核心考题 | 文件链接 | 考察重点与关键口述要点 |
| :---: | :--- | :--- | :--- |
| **01** | **多头注意力 (MHA)** | [`01_multi_head_attention.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/01_multi_head_attention.py) | 4D 张量 `[B, H, S, d_k]` 并行计算、缩放因子 $\frac{1}{\sqrt{d_k}}$、`.contiguous()` 内存连续化 |
| **02** | **分组查询注意力 (GQA)** | [`02_grouped_query_attention.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/02_grouped_query_attention.py) | 现代大模型 (Llama 3 / Qwen) 标配；`repeat_interleave` 扩展 KV 头；节省推理显存 |
| **03** | **均方根归一化 (RMSNorm)** | [`03_rmsnorm.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/03_rmsnorm.py) | 舍弃均值中心化（假设均值接近0），减少一次全局同步访存，提升 20%~30% 训练速度 |
| **04** | **旋转位置编码 (RoPE)** | [`04_rope.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/04_rope.py) | 复数空间旋转矩阵；内积自生相对位置特性；外推性强 |
| **05** | **SwiGLU 前馈网络** | [`05_swiglu.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/05_swiglu.py) | 门控线性单元；$\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) W_{\text{down}}$；隐藏层 $\frac{8}{3}d$ |
| **06** | **带 KV Cache 的推理注意力** | [`06_kv_cache_attention.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/06_kv_cache_attention.py) | Prefill (全量计算) vs Decode (增量单 token 计算)；每步将计算量降至 $\mathcal{O}(S)$ |
| **07** | **低秩自适应 (LoRA)** | [`07_lora_linear.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/07_lora_linear.py) | $W_0 + \frac{\alpha}{r} B A$；B 必须全 0 初始化（确保训练初始无扰动）；冻结主权重 |
| **08** | **稳定 Softmax & 交叉熵** | [`08_stable_softmax_and_cross_entropy.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/08_stable_softmax_and_cross_entropy.py) | 减去最大值防指数溢出；Log-Sum-Exp 技巧；`gather` 快速提取真实标签 |
| **09** | **直接偏好优化损失 (DPO)** | [`09_dpo_loss.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/09_dpo_loss.py) | 跳过奖励模型与 PPO，直接用策略与参考模型胜负概率差优化；`F.logsigmoid` 防下溢 |
| **10** | **对比学习损失 (InfoNCE)** | [`10_infonce_loss.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/10_infonce_loss.py) | 特征 L2 归一化后点积即余弦相似度；对角线即正样本；巧妙利用对角线标签化为交叉熵 |
| **11** | **Adam 优化器更新步** | [`11_adam_step.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/11_adam_step.py) | 一阶动量 $m$ (方向) 与二阶动量 $v$ (尺度)；偏差修正 $\frac{1}{1-\beta^t}$ 消除初态偏 0 误差 |
| **12** | **两层 MLP 手写反向传播** | [`12_mlp_backpropagation.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/12_mlp_backpropagation.py) | 纯矩阵求导与链式法则；$dW_2 = A_1^T dZ_2$；ReLU 导数掩码传递 |
| **13** | **纯向量化 K-Means** | [`13_kmeans_vectorized.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/13_kmeans_vectorized.py) | 广播计算距离矩阵 `[N, 1, D] - [1, K, D]` 避免双重循环；向量化均值中心更新 |
| **14** | **主成分分析 (PCA via SVD)** | [`14_pca_svd.py`](file:///Users/peng/Documents/GitHub/Deep-ML-Solutions/interview_core/14_pca_svd.py) | 中心化；SVD 避免计算协方差矩阵导致的条件数平方（防止数值精度恶化）；取右奇异向量投影 |

---

## 💡 面试通用加分秘籍

1. **先标清楚张量形状 (Tensor Shapes)**：
   面试官看代码首先找输入输出形状。在变量赋值后随手加上 `# [B, S, D]`，瞬间体现专业度。
2. **切忌在 GPU 张量上写 Python `for` 循环**：
   凡是能用张量广播（Broadcasting）或者额外增加一维（如多头维度 `n_heads`）解决的问题，绝不写循环。
3. **遇到转置 `transpose` 马上想起 `.contiguous()`**：
   在需要紧接着做 `.view()` 变换前，主动提一句：“因为转置后内存步长不连续，这里需要先调用 `.contiguous()`”，面试官好感度拉满。
