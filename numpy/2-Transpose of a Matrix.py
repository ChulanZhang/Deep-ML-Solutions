def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    if not a:
        return []
    
    results = [[] for _ in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            results[j].append(a[i][j])
    return results