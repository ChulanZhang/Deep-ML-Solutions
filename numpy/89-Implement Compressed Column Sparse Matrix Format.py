import numpy as np

def compressed_col_sparse_matrix(dense_matrix: list[list[float]]) -> tuple[list[float], list[int], list[int]]:
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.
    """
    if not dense_matrix or not dense_matrix[0]:
        return [], [], [0]
        
    cols = len(dense_matrix[0])
    rows = len(dense_matrix)
    
    values = []
    row_indices = []
    col_ptr = [0]
    
    num_non_zero = 0
    for j in range(cols):
        for i in range(rows):
            val = dense_matrix[i][j]
            if val != 0:
                values.append(val)
                row_indices.append(i)
                num_non_zero += 1
        col_ptr.append(num_non_zero)
        
    return values, row_indices, col_ptr
