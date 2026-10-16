# PyTorch 审计报告

## 结论与边界

本轮完成全量源码盘点、题号核对和公开样例回放，**重写 40 道大模型相关核心题，整理原 14 个 interview 模块，新增 1 个组合 decoder**。另补充检查并归档 6 个嵌套 llm_blocks 模块。

**不是“全部历史解法已修复”。** 未进入新 pytorch/ 的旧解法只归档，不作为推荐答案。NumPy 版本未修复。没有向 Deep-ML 网站提交答案，没有隐藏判题结果。

原始提交：c201923f74cd181fb16259ff2cf42754334bf2e3。核对日期：2026-10-15（America/Indiana/Indianapolis；UTC 抓取记录为 10 月 16 日）。官网资料通过公开 GET API 获取，来源/时间/哈希保存在 audit/public_examples.json。没有读取账户或调用提交接口。

## 范围与实测结果

| 项目 | 实际结果 |
|---|---|
| 原顶层 PyTorch 题解 | 248 个；题号映射和公开回放以这些文件为范围 |
| 原 interview 模块 | 14 个 |
| 嵌套 llm_blocks | 6 个，单独检查，不强配官网号 |
| 语法解析与编译 | 268 / 268 通过 |
| 独立导入 | 12 个顶层文件因缺少 torch 导入报错 |
| 静态可疑标记 | 80 个文件使用未导入的 np；75 个 torch.round 位置参数；4 个 NumPy 风格 transpose；11 个疑似整数 torch.sqrt |
| 至少一个静态标记 | 126 个文件；类别重叠，标记不等于已确认的语义错误 |
| 原稿题号映射 | 245 个文件对应 218 个官网 ID；39 个数字前缀需要更正；27 个重复别名 |
| 未硬配官网号 | 3 个自定义项目/练习 |
| 新规范索引 | 加入正确的 #125 Sparse MoE 后，共 219 个官网 ID，各一行 |
| 新核心实现 | 40 道独立单题，文件名与官网 ID 对应 |
| 面试模块 | 15 个；01…15 是学习顺序，不是官网 ID |

### 原 248 个文件的公开样例回放

| 文件状态 | 数量 |
|---|---:|
| 有效样例均通过 | 30 |
| 有有效样例失败 | 198 |
| 样例没有调用原实现 | 7 |
| 没有公开 PyTorch 样例 | 10 |
| 自定义练习未匹配 | 3 |

失败包含运行错误、接口不匹配和输出不匹配，不能一概称为算法错误。按样例计数：142 次有效通过、761 次运行错误、37 次输出不匹配、29 次未调用被测实现。逐条结果见 audit/public_original.json。没有公开样例不等于网站没有隐藏测试。

### 新核心版本

40 道题共捕获 174 个公开样例：**167 个实际调用实现并通过**；7 个未调用被测实现，不计入正确性证明。#374 的 6 个样例重定义了 CharTokenizer；#94 的第 5 个样例没有调用答案。#53 部分样例覆盖 compute_qkv，因此另加投影验证。

**108 个独立 pytest 测试通过，0 失败**，覆盖 40 个单题和 15 个 interview 模块，包含 SDPA 数值对照、梯度、gradcheck、因果性、缓存一致性和边界条件。

原 14 个 interview 模块对应的 21 项回归测试为 12 通过、9 失败；修订后通过。记录见 audit/pytest_interview_original.xml。

本地环境：Python 3.13.5、PyTorch 2.10.0+cpu、NumPy 2.3.5、pytest 9.0.2。CPU；广泛测试 float32/float64，仅对选定归一化/掩码边界测试 fp16/bfloat16。没有 GPU、速度或显存 benchmark。GitHub Actions 安装同一依赖版本，并重新生成执行报告。

### 嵌套 llm_blocks 的补充检查

6 个嵌套模块均能解析、导入；11 项功能/精度检查为 **6 通过、5 失败**，见 audit/nested_legacy.json。独立 RoPE、SwiGLU 和无 RoPE 的 GQA 参考计算通过；但 GQA 中应用 RoPE 的代码被注释，TransformerBlock 和 LlamaModel 因而忽略旋转。把旋转全部改为恒等旋转，旧 Llama 输出完全不变。

旧 Llama 的多 token fp16 前向存在 Float/Half 类型不一致。RMSNorm 强制转 float32 会丢失 float64 精度，这是支持边界，不等于 float32 公式错误。旧 Llama 本来就没有 KV cache 实现，start_pos 参数不代表缓存能力。

这些历史模块没有原地修复，保留在 archive/pytorch_legacy/llm_blocks/；推荐学习已验证的 interview_core/15_decoder_block_with_cache.py。全部 268 个原 Python 源码按 SHA256 校验保留。

## 关键修订

| 模块 | 修订与验证 |
|---|---|
| #53 / #94 / #107 | 修复标量 torch.sqrt 与 NumPy 风格 transpose；对照 PyTorch SDPA |
| #94 / #391 | 正确拆分/合并 head，检查头数可整除，数值及梯度验证 |
| #109 | 最后一维 population variance，对照 F.layer_norm |
| #156 | 本题 first_half * SiLU(second_half)，不混用 gate/value 顺序 |
| #178 | 当前 PyTorch 接口为 FFNBlock，保留 linear1 / linear2 / dropout |
| #229 | gather 所选 expert_outputs，再加权求和，而不只返回门控索引 |
| #222 / #223 | 保持 Tensor 梯度图；遵循题目的 A/B 和 dequant 约定 |
| #372 / interview RMSNorm | 低精度平方均值用 float32 累加，保留 float64 输入精度 |
| interview MHA | 不用 fp16 无法表示的 -1e9；用 -inf 并显式处理全屏蔽行 |
| interview KV Cache | prefill / chunk 使用 past_length 构造 causal mask；full/step/chunk 对照 |
| interview K-means | 正确提交最后一次中心更新，返回与最终中心对应的 labels |
| 组合 decoder | GQA + RoPE + KV Cache + RMSNorm + SwiGLU 联动；检查缓存头数和位置偏移 |

## 官网接口本身的差异

#178 的共用题面使用函数 ffn，PyTorch starter/tests 要求 FFNBlock。

#222 本题 B=(in,r)、A=(r,out)，计算 x@B@A；interview 版本采用 A=(r,in)、B=(out,r)，不能混用命名与形状。#223 本题指定 W=q*scale+zero_point，不是通用 `(q-zp)*scale`；它不是完整 NF4 QLoRA。

#156 指定第二半经过 SiLU。旧开源题库 metadata 与当前官网编号有差异，不能只查旧 GitHub 目录。

#409 共用 NumPy 描述要求同分选高索引，但 PyTorch 公开样例 #3/#5 要求低索引。本版采用低索引优先，记录歧义，不能承诺隐藏判题完全一致。#419 共用说明提到 NumPy RNG，PyTorch 样例使用 PyTorch 的采样结果；本版使用局部 torch.Generator。

#294 的半整数舍入依赖 dtype，本版明确使用 PyTorch 样例采用的 float32。#388 是双向 sliding-window radius，不是 causal decoder window。#384 是 SimCLR/NT-Xent；interview_core/10 是 CLIP 对称跨视图损失，不是同一公式。

旧 Calculate Number of Parameters 同名题在官网 #291 与 #371 都存在。本次规范到 #291，并保留 #371 作为别名歧义，不能据题名声称唯一确定历史来源。

## 测试协议与复现

公开回放在新 namespace 中加载解法、运行捕获的公开测试并记录 stdout。比较保留嵌套和非数字 token，忽略空白/逗号；数字 atol=5.1e-5、rtol=1e-5。每例 5 秒 POSIX 超时。通过调用跟踪识别测试是否真的执行解法，并记录被测试文本覆盖的定义。

这不是官方 judge comparator。独立 pytest 使用更严格的数值/梯度/不变量对照。

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-review.txt
python -m pytest tests -q
python audit/run_public_tests.py --source verified --output /tmp/core-public.json
python audit/run_public_tests.py --source original --output /tmp/original-public.json
python audit/check_nested_legacy.py
python audit/validate_manifest.py
```

original 读取 archive/pytorch_legacy，verified 读取新 pytorch。pytest.ini 仅收集 tests/。历史 numpy/tests 的 10 个失效路径收集错误没有修复，也不属于 108 passed。

## 未覆盖范围

40 道以外的旧解法没有批量修复；NumPy、CUDA/Triton、分页 cache、复杂 padding/packed sequences、完整 NF4 训练、GPU 性能均未验证。数学参考 FlashAttention 不等于高性能融合 kernel。

## 来源

官网目录：https://api.deep-ml.com/list-problems

单题：https://api.deep-ml.com/fetch-problem?problem_id=94 （其余 URL、时间、哈希见快照）

官网 FAQ：https://www.deep-ml.com/faq

官方旧开源库：https://github.com/Open-Deep-ML/DML-OpenProblem ，本次参考 commit 04708b485a051316c25b91e4cb9a2ada0ccf7fbf；当前题号以官网 API 为准。

PyTorch：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

公开题名、starter、样例归原作者所有；解法、独立测试与复习说明重新编写，保留来源供个人复习与复现。
