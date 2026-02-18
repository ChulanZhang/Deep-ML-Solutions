# 贡献指南 / Contributing Guide

感谢你对本项目的关注！这个仓库主要用于记录个人在 Deep-ML 上的刷题进度。

Thank you for your interest! This repository is primarily for tracking personal progress on Deep-ML problems.

## 如何组织题解 / How to Organize Solutions

### 1. 创建题目文件夹 / Create Problem Folder

在对应难度的目录下创建新文件夹：

```
solutions/[difficulty]/[problem-id]-[problem-name]/
```

例如 / Example:
```
solutions/easy/001-matrix-transpose/
solutions/medium/015-neural-network-forward/
```

### 2. 使用模板 / Use Template

复制 `solutions/template.md` 作为起点：

```bash
cp solutions/template.md solutions/easy/001-your-problem/README.md
```

### 3. 添加代码文件 / Add Code Files

```
solutions/easy/001-your-problem/
├── README.md          # 题解说明
├── solution.py        # Python 实现
├── solution.js        # JavaScript 实现（可选）
└── test.py           # 测试文件（可选）
```

### 4. 更新索引 / Update Index

在对应难度的 README.md 中添加题目链接。

Add the problem link in the corresponding difficulty's README.md.

## 代码风格 / Code Style

### Python

- 使用 4 个空格缩进 / Use 4 spaces for indentation
- 遵循 PEP 8 规范 / Follow PEP 8 guidelines
- 添加适当的注释和文档字符串 / Add appropriate comments and docstrings
- 包含测试用例 / Include test cases

### 文档 / Documentation

- 使用中英双语 / Use bilingual (Chinese/English)
- 清晰的问题描述 / Clear problem description
- 详细的解题思路 / Detailed approach explanation
- 复杂度分析 / Complexity analysis
- 测试用例 / Test cases

## 提交规范 / Commit Guidelines

提交信息格式 / Commit message format:

```
[类型] 简短描述

详细说明（可选）
```

类型 / Types:
- `[新增]` / `[Add]` - 添加新题解
- `[更新]` / `[Update]` - 更新现有题解
- `[修复]` / `[Fix]` - 修复错误
- `[文档]` / `[Docs]` - 文档更新
- `[重构]` / `[Refactor]` - 代码重构

示例 / Examples:
```
[新增] 添加矩阵转置题解 (Easy #001)
[Add] Add matrix transpose solution (Easy #001)

[更新] 优化神经网络前向传播算法
[Update] Optimize neural network forward propagation

[文档] 更新 README 和进度追踪
[Docs] Update README and progress tracking
```

## 学习笔记 / Learning Notes

在 `notes/` 目录下可以添加：
- 知识点总结 / Concept summaries
- 算法模板 / Algorithm templates
- 常见错误 / Common mistakes
- 优化技巧 / Optimization tips

## 问题和建议 / Questions and Suggestions

如果你有任何问题或建议，欢迎：
- 提交 Issue
- 发起 Discussion
- 提交 Pull Request

If you have questions or suggestions:
- Submit an Issue
- Start a Discussion
- Submit a Pull Request

---

Happy coding! 加油刷题！💪
