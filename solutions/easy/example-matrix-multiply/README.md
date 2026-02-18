# [示例] 矩阵乘法 / Example: Matrix Multiplication

## 问题描述 / Problem Description

实现一个简单的矩阵乘法函数。给定两个矩阵 A (m×n) 和 B (n×p)，返回它们的乘积矩阵 C (m×p)。

Implement a simple matrix multiplication function. Given two matrices A (m×n) and B (n×p), return their product matrix C (m×p).

**难度 / Difficulty**: Easy

**标签 / Tags**: 线性代数 (Linear Algebra), 矩阵运算 (Matrix Operations)

## 解题思路 / Approach

### 中文思路

1. 检查矩阵维度是否匹配（A 的列数必须等于 B 的行数）
2. 初始化结果矩阵 C，大小为 m×p
3. 使用三重循环计算矩阵乘法：
   - 外层循环遍历 A 的行
   - 中层循环遍历 B 的列
   - 内层循环计算点积

### English Approach

1. Verify matrix dimensions are compatible (columns of A must equal rows of B)
2. Initialize result matrix C with size m×p
3. Use triple nested loops for matrix multiplication:
   - Outer loop iterates through rows of A
   - Middle loop iterates through columns of B
   - Inner loop computes dot product

## 代码实现 / Implementation

### Python - 基础实现

```python
def matrix_multiply(A, B):
    """
    矩阵乘法
    
    Args:
        A: list of lists, shape (m, n)
        B: list of lists, shape (n, p)
    
    Returns:
        C: list of lists, shape (m, p)
    """
    # 获取矩阵维度
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # 检查维度
    if len(B) != n:
        raise ValueError("矩阵维度不匹配 / Matrix dimensions incompatible")
    
    # 初始化结果矩阵
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    # 矩阵乘法
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C
```

### Python - NumPy 实现

```python
import numpy as np

def matrix_multiply_numpy(A, B):
    """
    使用 NumPy 进行矩阵乘法
    """
    return np.matmul(A, B).tolist()
    # 或者使用: np.dot(A, B) 或 A @ B
```

## 复杂度分析 / Complexity Analysis

- **时间复杂度 / Time Complexity**: O(m × n × p)
  - 三重循环，分别遍历 m 行、p 列和 n 个元素求和
- **空间复杂度 / Space Complexity**: O(m × p)
  - 需要存储结果矩阵

## 测试用例 / Test Cases

```python
# Test Case 1: 基本矩阵乘法
A1 = [[1, 2], [3, 4]]
B1 = [[5, 6], [7, 8]]
expected1 = [[19, 22], [43, 50]]
assert matrix_multiply(A1, B1) == expected1
print("Test 1 passed!")

# Test Case 2: 不同维度
A2 = [[1, 2, 3]]
B2 = [[4], [5], [6]]
expected2 = [[32]]
assert matrix_multiply(A2, B2) == expected2
print("Test 2 passed!")

# Test Case 3: 单位矩阵
A3 = [[1, 0], [0, 1]]
B3 = [[5, 6], [7, 8]]
expected3 = [[5, 6], [7, 8]]
assert matrix_multiply(A3, B3) == expected3
print("Test 3 passed!")
```

## 知识点 / Key Concepts

- **矩阵乘法规则 / Matrix Multiplication Rules**
  - 矩阵 A (m×n) 和 B (n×p) 可以相乘
  - 结果矩阵 C 的维度为 (m×p)
  - C[i][j] = Σ(A[i][k] * B[k][j]) for k from 0 to n-1

- **优化方法 / Optimization Methods**
  - 使用 NumPy 等优化库（底层使用 BLAS）
  - 缓存友好的访问模式
  - Strassen 算法（用于大矩阵）

## 参考资料 / References

- [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- [Matrix Multiplication - Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication)

## 笔记 / Notes

这是一个经典的线性代数问题，是许多机器学习算法的基础。在实际应用中：
- 神经网络的前向传播大量使用矩阵乘法
- 建议使用优化库（如 NumPy）而不是手动实现
- 注意矩阵维度的匹配

This is a classic linear algebra problem fundamental to many ML algorithms. In practice:
- Neural network forward propagation heavily uses matrix multiplication
- Use optimized libraries (like NumPy) instead of manual implementation
- Pay attention to matrix dimension compatibility
