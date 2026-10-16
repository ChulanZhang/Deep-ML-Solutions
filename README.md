# Deep-ML Solutions — PyTorch / LLM Review

以大模型核心模块为主的 PyTorch coding 复习仓库。官网题号与学习顺序分开，答案与测试状态分开。

**本轮重写 40 道核心题，不代表全部历史解法已修复。** 原 248 个顶层 PyTorch 文件、6 个嵌套 llm_blocks 文件保留在归档中；NumPy 目录尚未经过本轮修复。

## 从哪里开始

| 入口 | 内容 |
|---|---|
| [复习清单](REVIEW_CHECKLIST.md) | P0/P1/P2 顺序、每题掌握标准、组合 decoder 练习 |
| [pytorch/](pytorch/) | 40 道按真实官网 ID 命名的独立单题实现 |
| [interview_core/](interview_core/) | 15 个面试模块；前缀 01…15 只是学习顺序，不是官网号 |
| [完整题号索引](Deep_ML_Problems_List.md) | 219 个官网 ID，各一行；保留旧文件映射和未修复标记 |
| [审计报告](AUDIT_REPORT.md) | 实测结果、发现的问题、官网接口差异与未覆盖范围 |
| [归档](archive/) | 历史答案与旧清单，仅供追溯，不作为默认答案 |

## 已执行的验证

CPU / PyTorch 2.10.0：**108 个独立测试通过**；**167 个确实调用实现的官网公开样例通过**。另 7 个样例没有调用被测实现，明确排除，不能算正确性证明。

没有向 Deep-ML 网站提交，没有隐藏判题结果，没有 GPU 性能结论。公开样例与测试执行报告见 [audit/](audit/)。

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-review.txt
python -m pytest tests -q
python audit/run_public_tests.py --source verified --output /tmp/core-public.json
python audit/validate_manifest.py
```

GitHub Actions 使用同一套独立测试和固定的公开样例快照。默认 pytest 仅运行新的 tests/，不包含尚未修复的 numpy/tests。

## 最先复习

`#53 → #94 → #107 → #376 → #391 → #381 → #109/#372 → #156/#178 → #408`。

然后自己组合一个带 GQA、RoPE、KV Cache 和 SwiGLU 的 decoder block；不要只记住不带 batch 的单题代码。
