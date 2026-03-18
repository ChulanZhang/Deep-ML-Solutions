def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    """
    Calculate the inverse of a 2x2 matrix.

    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]

    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # 提取 2x2 矩阵的元素
    a, b = matrix[0]
    c, d = matrix[1]

    # 1. 计算行列式 (Determinant): ad - bc
    det = a * d - b * c

    # 2. 如果行列式为 0，说明矩阵是奇异矩阵，不可逆，返回 None
    if det == 0:
        return None

    # 3. 应用 2x2 矩阵求逆公式: 
    # A^-1 = (1/det) * [[d, -b], [-c, a]]
    inverse_matrix = [
        [d / det, -b / det],
        [-c / det, a / det]
    ]

    return inverse_matrix
