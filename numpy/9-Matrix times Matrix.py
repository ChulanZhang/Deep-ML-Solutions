def matrixmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """
    Multiply two matrices A and B. Return -1 if they cannot be multiplied.
    
    Args:
        a: Matrix A of shape (n, m)
        b: Matrix B of shape (m, k)
        
    Returns:
        Resulting matrix C of shape (n, k), or -1 if invalid dimensions.
    """
    # 1. 获取矩阵的维度
    # a 有 n 行，m 列；b 有 p 行，k 列
    n = len(a)
    m = len(a[0]) if n > 0 else 0
    p = len(b)
    k = len(b[0]) if p > 0 else 0

    # 2. 检查两者是否可以直接相乘: A 的列数 (m) 必须等于 B 的行数 (p)
    if m != p:
        return -1

    # 初始化结果矩阵 C (n 行 k 列)，全部填 0
    c = [[0 for _ in range(k)] for _ in range(n)]

    # 3. 矩阵乘法核心逻辑: 三重循环
    # 结果矩阵 C 的第 i 行 第 j 列的元素等于 A 的第 i 行与 B 的第 j 列的点积
    for i in range(n):              # 遍历 A 的行
        for j in range(k):          # 遍历 B 的列
            for l in range(m):      # 遍历 A 的列 / B 的行，进行点积累加
                c[i][j] += a[i][l] * b[l][j]

    return c
