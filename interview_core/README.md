# Interview modules

这里的 01…15 是学习顺序，不是 Deep-ML 官网编号。单题接口见 ../pytorch/；以下模块用于练习 batch、nn.Module、梯度与组合。

| 顺序 | 模块 | 相关官网练习 |
|---|---|---|
| 01 | [01_multi_head_attention](01_multi_head_attention.py) | #94 / #107 |
| 02 | [02_grouped_query_attention](02_grouped_query_attention.py) | #391 |
| 03 | [03_rmsnorm](03_rmsnorm.py) | #372 |
| 04 | [04_rope](04_rope.py) | #381；此文件用 half-split，本题用 adjacent pairs |
| 05 | [05_swiglu](05_swiglu.py) | #156 / #178 |
| 06 | [06_kv_cache_attention](06_kv_cache_attention.py) | #376；扩展到多头 causal prefill/chunk |
| 07 | [07_lora_linear](07_lora_linear.py) | #222；A/B 命名和形状不同 |
| 08 | [08_stable_softmax_and_cross_entropy](08_stable_softmax_and_cross_entropy.py) | #23 / #39；训练版不 round |
| 09 | [09_dpo_loss](09_dpo_loss.py) | #382；返回 Tensor loss |
| 10 | [10_infonce_loss](10_infonce_loss.py) | #384 仅概念相关；这里是 CLIP，不是 SimCLR |
| 11 | [11_adam_step](11_adam_step.py) | #49 |
| 12 | [12_mlp_backpropagation](12_mlp_backpropagation.py) | 自定义手写反向传播练习 |
| 13 | [13_kmeans_vectorized](13_kmeans_vectorized.py) | #17，仅概念相关、初始化接口不同 |
| 14 | [14_pca_svd](14_pca_svd.py) | #19，返回契约不同 |
| 15 | [15_decoder_block_with_cache](15_decoder_block_with_cache.py) | 组合练习，不单独硬配官网题号 |

Mask 约定：MHA/GQA 模块的 bool 或 0/1 mask 中 True/1 表示允许；不要直接套用 nn.MultiheadAttention 的相反布尔语义。KV Cache 与组合 decoder 会自行构造 causal mask。

KV Cache 模块返回的 cache 保留计算图；推理用 torch.no_grad()，本轮没有实现分页缓存、滑动淘汰或生产调度。

15_decoder_block_with_cache 已测试 full prefill、逐 token 和多 token chunk 一致性，缓存存 Hkv 个头，位置从 past_length 开始。
