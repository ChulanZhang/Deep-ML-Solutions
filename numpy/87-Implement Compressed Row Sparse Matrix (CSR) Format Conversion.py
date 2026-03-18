import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    values = []
    col_indices = []
    row_ptr = [0]
    
    num_non_zero = 0
    for row in dense_matrix:
        for j, val in enumerate(row):
            if val != 0:
                values.append(val)
                col_indices.append(j)
                num_non_zero += 1
        row_ptr.append(num_non_zero)
        
    return (values, col_indices, row_ptr)
