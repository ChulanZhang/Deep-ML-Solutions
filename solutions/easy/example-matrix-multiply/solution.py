"""
矩阵乘法示例
Matrix Multiplication Example
"""

def matrix_multiply(A, B):
    """
    矩阵乘法
    
    Args:
        A: list of lists, shape (m, n)
        B: list of lists, shape (n, p)
    
    Returns:
        C: list of lists, shape (m, p)
    """
    # 验证矩阵非空
    if not A or not B or not A[0] or not B[0]:
        raise ValueError("矩阵不能为空 / Matrices cannot be empty")
    
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


def test_matrix_multiply():
    """测试函数"""
    # Test Case 1: 基本矩阵乘法
    A1 = [[1, 2], [3, 4]]
    B1 = [[5, 6], [7, 8]]
    expected1 = [[19, 22], [43, 50]]
    assert matrix_multiply(A1, B1) == expected1
    print("✓ Test 1 passed!")

    # Test Case 2: 不同维度
    A2 = [[1, 2, 3]]
    B2 = [[4], [5], [6]]
    expected2 = [[32]]
    assert matrix_multiply(A2, B2) == expected2
    print("✓ Test 2 passed!")

    # Test Case 3: 单位矩阵
    A3 = [[1, 0], [0, 1]]
    B3 = [[5, 6], [7, 8]]
    expected3 = [[5, 6], [7, 8]]
    assert matrix_multiply(A3, B3) == expected3
    print("✓ Test 3 passed!")
    
    # Test Case 4: 空矩阵验证
    try:
        matrix_multiply([], [[1]])
        assert False, "Should raise ValueError for empty matrix"
    except ValueError as e:
        print("✓ Test 4 passed! (Empty matrix validation)")
    
    print("\n所有测试通过！/ All tests passed!")


if __name__ == "__main__":
    test_matrix_multiply()
