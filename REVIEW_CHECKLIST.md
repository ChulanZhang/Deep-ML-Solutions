# 大模型 PyTorch Coding 复习清单

先练 P0 的注意力与解码，再练 P1 的 MoE、微调、采样与损失。P2 用于扩展，不建议按题号从头刷。下面所有数字均为官网 ID；四位文件名前缀只是补零。

每题验收：不看答案写出代码，标清每一步 shape，解释一个边界条件，运行数值测试而非只打印 shape。勾选框记录你的掌握程度，不代表代码审计状态。

## P0：注意力与自回归解码主线

| 复习 | 官网题目 | 必须掌握 | 修订代码 |
|---|---|---|---|
| [ ] | [#23 Softmax](https://www.deep-ml.com/problems/23) | subtract-max；dim/keepdim；大 logits 不溢出 | [代码](pytorch/0023_softmax.py) |
| [ ] | [#39 Log Softmax](https://www.deep-ml.com/problems/39) | logsumexp；不先 softmax 再 log；训练计算不 round | [代码](pytorch/0039_log_softmax.py) |
| [ ] | [#53 Self-Attention](https://www.deep-ml.com/problems/53) | Q/K/V 投影；(S,d)→(S,S)→(S,d_v) | [代码](pytorch/0053_self_attention.py) |
| [ ] | [#94 Multi-Head Attention](https://www.deep-ml.com/problems/94) | reshape/transpose/合并 heads；缩放用 head_dim | [代码](pytorch/0094_multi_head_attention.py) |
| [ ] | [#107 Masked Self-Attention](https://www.deep-ml.com/problems/107) | additive/bool mask；True 的语义；全屏蔽行 | [代码](pytorch/0107_masked_attention.py) |
| [ ] | [#376 KV Cache](https://www.deep-ml.com/problems/376) | 追加 K/V；缓存按步增长；与 full causal 输出对照 | [代码](pytorch/0376_kv_cache_step.py) |
| [ ] | [#391 Grouped Query Attention](https://www.deep-ml.com/problems/391) | Hq/Hkv；头数整除；MHA/MQA 两个特例 | [代码](pytorch/0391_grouped_query_attention.py) |
| [ ] | [#381 RoPE](https://www.deep-ml.com/problems/381) | 相邻二维旋转；只旋转 Q/K；cache position offset | [代码](pytorch/0381_rope.py) |
| [ ] | [#109 LayerNorm](https://www.deep-ml.com/problems/109) | 最后一维；population variance；不能误用样本方差 | [代码](pytorch/0109_layer_norm.py) |
| [ ] | [#372 RMSNorm](https://www.deep-ml.com/problems/372) | 不减均值；低精度用 float32 累加；保留梯度 | [代码](pytorch/0372_rmsnorm.py) |
| [ ] | [#156 SwiGLU](https://www.deep-ml.com/problems/156) | 本题 first_half * SiLU(second_half)；gate/value 顺序 | [代码](pytorch/0156_swiglu.py) |
| [ ] | [#178 FFN + Residual + Dropout](https://www.deep-ml.com/problems/178) | PyTorch 类 FFNBlock；linear1/linear2/dropout；train/eval | [代码](pytorch/0178_ffn_block.py) |
| [ ] | [#408 Pre-Norm vs Post-Norm](https://www.deep-ml.com/problems/408) | 归一化与 residual 顺序；本题子层是线性占位 | [代码](pytorch/0408_pre_post_norm.py) |

## P1：MoE、微调、采样与训练目标

| 复习 | 官网题目 | 必须掌握 | 修订代码 |
|---|---|---|---|
| [ ] | [#124 Noisy Top-K Gating](https://www.deep-ml.com/problems/124) | softplus 噪声尺度、输入噪声 N、top-k 门控 | [代码](pytorch/0124_noisy_topk_gating.py) |
| [ ] | [#229 Sparse MoE Routing](https://www.deep-ml.com/problems/229) | gather 选中专家输出，重新归一化，返回加权结果 | [代码](pytorch/0229_moe_topk_routing.py) |
| [ ] | [#125 Sparse Mixture of Experts](https://www.deep-ml.com/problems/125) | 每 token 选 k 个专家，执行并组合线性变换 | [代码](pytorch/0125_sparse_moe.py) |
| [ ] | [#409 Shared-Expert MoE](https://www.deep-ml.com/problems/409) | shared 始终参与；routed 重归一化；同分规则 | [代码](pytorch/0409_moe_shared_expert.py) |
| [ ] | [#222 LoRA Forward](https://www.deep-ml.com/problems/222) | 本题 x@B@A，B=(in,r)、A=(r,out)；alpha/r、零更新 | [代码](pytorch/0222_lora_forward.py) |
| [ ] | [#223 QLoRA Forward](https://www.deep-ml.com/problems/223) | 本题 q*scale+zero_point；仅为简化前向，不是完整 NF4 系统 | [代码](pytorch/0223_qlora_forward.py) |
| [ ] | [#383 Top-p Sampling](https://www.deep-ml.com/problems/383) | 保留首次跨过 p 的 token；stable ties；重新归一化 | [代码](pytorch/0383_top_p.py) |
| [ ] | [#419 Combined Sampling](https://www.deep-ml.com/problems/419) | temperature→top-k→softmax→top-p→sample；贪心与局部 RNG | [代码](pytorch/0419_combined_sampling.py) |
| [ ] | [#320 Perplexity](https://www.deep-ml.com/problems/320) | exp(-mean(log p))，不是 mean(1/p)；评估忽略 padding | [代码](pytorch/0320_perplexity.py) |
| [ ] | [#382 DPO Loss](https://www.deep-ml.com/problems/382) | policy/ref chosen/rejected log-ratio；logsigmoid；报告与训练 loss 分开 | [代码](pytorch/0382_dpo_loss.py) |
| [ ] | [#384 InfoNCE / SimCLR](https://www.deep-ml.com/problems/384) | 相邻正对；排除自身；同视图负例；与 CLIP N×N loss 区分 | [代码](pytorch/0384_infonce.py) |

## P2：基础、效率与部署扩展

| 复习 | 官网题目 | 必须掌握 | 修订代码 |
|---|---|---|---|
| [ ] | [#49 Adam](https://www.deep-ml.com/problems/49) | 一/二阶动量、t 从 1 开始、bias correction；区分 AdamW | [代码](pytorch/0049_adam.py) |
| [ ] | [#85 Sinusoidal Position Encoding](https://www.deep-ml.com/problems/85) | sin/cos；奇数 d_model；本题输出 float16 | [代码](pytorch/0085_sinusoidal_position.py) |
| [ ] | [#101 GRPO Objective](https://www.deep-ml.com/problems/101) | clipped surrogate；本题 KL 乘 rho；objective 与 loss 符号 | [代码](pytorch/0101_grpo_objective.py) |
| [ ] | [#151 Dropout](https://www.deep-ml.com/problems/151) | inverted dropout；train/eval；backward 重用 mask | [代码](pytorch/0151_dropout.py) |
| [ ] | [#224 Group Relative Advantage](https://www.deep-ml.com/problems/224) | 组内标准化、零方差、population std | [代码](pytorch/0224_group_relative_advantage.py) |
| [ ] | [#225 KL Estimator](https://www.deep-ml.com/problems/225) | r=pi_ref/pi_theta；expm1(log r)-log r | [代码](pytorch/0225_kl_estimator.py) |
| [ ] | [#263 Binary Cross-Entropy](https://www.deep-ml.com/problems/263) | probability/logits 两类接口；float32 中 epsilon 的表示精度 | [代码](pytorch/0263_binary_cross_entropy.py) |
| [ ] | [#294 INT8 Quantization](https://www.deep-ml.com/problems/294) | [-127,127]；全零 scale；半整数舍入与 dtype | [代码](pytorch/0294_int8_quantization.py) |
| [ ] | [#374 Character Tokenizer](https://www.deep-ml.com/problems/374) | stoi/itos/BOS/EOS；LongTensor/Embedding；独立验证 | [代码](pytorch/0374_char_tokenizer.py) |
| [ ] | [#375 Learned Positions](https://www.deep-ml.com/problems/375) | start_pos 切表、batch 广播、边界、梯度 | [代码](pytorch/0375_learned_position.py) |
| [ ] | [#390 Multiquery Attention](https://www.deep-ml.com/problems/390) | 多个 Q，共享 K/V，输出投影 | [代码](pytorch/0390_multiquery_attention.py) |
| [ ] | [#407 QK-Norm](https://www.deep-ml.com/problems/407) | L2 normalize 后除 temperature；不再除 sqrt(d) | [代码](pytorch/0407_qk_norm_attention.py) |
| [ ] | [#418 KV Cache Size](https://www.deep-ml.com/problems/418) | 2×L×B×S×Hkv×Dhead×bytes；注意不是 Hq | [代码](pytorch/0418_kv_cache_size.py) |
| [ ] | [#429 Quantization Quality](https://www.deep-ml.com/problems/429) | 同一批 token 的 perplexity delta；阈值非普适保证 | [代码](pytorch/0429_quantization_quality.py) |
| [ ] | [#208 Flash Attention Forward](https://www.deep-ml.com/problems/208) | online softmax 的 running max/sum/numerator；尾块 | [代码](pytorch/0208_flash_attention_reference.py) |
| [ ] | [#388 Sliding Window Attention](https://www.deep-ml.com/problems/388) | 本题双向半径，不是总宽度，也不默认 causal | [代码](pytorch/0388_sliding_window_attention.py) |

## 最重要的组合练习

**从零写一个 pre-norm decoder block：RMSNorm → GQA + RoPE + causal KV Cache → residual → RMSNorm → SwiGLU → residual。**

这是组合练习，不冒充单独官网题号。参考：[15_decoder_block_with_cache.py](interview_core/15_decoder_block_with_cache.py)。

验收时必须对照整段 prefill、逐 token decode、多个 token 一块 decode。缓存存 Hkv 而非 Hq 个头；缓存里的 K 已经旋转，不能二次旋转；chunk 的 query 绝对位置必须加 past_length。检查位置不是只检查输出 shape。

## 七个复习单元

| 单元 | 主线 | 验收 |
|---|---|---|
| 1 | #23/#39、#109/#372 | 不查资料写稳定 softmax/CE 和两种 normalization |
| 2 | #53→#94→#107 | 画全部 shape；说明 causal、padding、全屏蔽行 |
| 3 | #376→#391→#381 | 单步/分块 cache 与 full causal 一致 |
| 4 | #156→#178→#408→组合 decoder | 把 attention、RoPE、cache、FFN 正确接起来 |
| 5 | #124→#229→#125→#409 | top-k、权重、gather/scatter、shared expert |
| 6 | #222/#223、#382/#384 | LoRA 梯度与形状；DPO、CLIP、SimCLR 不混淆 |
| 7 | #383/#419、#208/#418、#320/#429 | 采样、online softmax、KV 显存量级、量化指标 |

interview_core/01…15 是学习顺序，不是 Deep-ML 题号。单题版遵循官网接口；模块版补 batch、nn.Module、梯度与组合。#408 不是完整 decoder，#223 不是完整 NF4 训练，#208 的数学参考不是高性能 CUDA kernel。

[完整题号索引](Deep_ML_Problems_List.md) · [测试与未覆盖范围](AUDIT_REPORT.md)。
