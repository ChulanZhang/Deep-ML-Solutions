def get_minor(matrix, i, j):
    """
    Get the minor of a matrix by removing the i-th row and j-th column.
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def determinant_recursive(matrix):
    """
    Recursive determinant calculation using Laplace expansion.
    """
    n = len(matrix)
    if n == 1:
        return float(matrix[0][0])
    if n == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
        
    det = 0.0
    # Expand along the first row
    for j in range(n):
        sign = (-1) ** j
        sub_det = determinant_recursive(get_minor(matrix, 0, j))
        det += sign * matrix[0][j] * sub_det
        
    return float(det)

def determinant_4x4(matrix: list[list[float]]) -> float:
    """
    Calculate the determinant of a 4x4 matrix using Laplace's Expansion method.
    """
    return determinant_recursive(matrix)
