import numpy as np

def matrix_image(A):
    """
    Find the column space of a matrix.
    Returns the basis vectors that span the column space (independent columns of A).
    """
    A_np = np.array(A)
    independent_cols = []
    current_rank = 0
    basis_indices = []
    
    for j in range(A_np.shape[1]):
        col = A_np[:, j:j+1]
        if not independent_cols:
            test_matrix = col
        else:
            # Combine current independent columns with the new column
            test_matrix = np.hstack(independent_cols + [col])
            
        new_rank = np.linalg.matrix_rank(test_matrix)
        
        # If the rank increases, it's an independent column (pivot column)
        if new_rank > current_rank:
            independent_cols.append(col)
            basis_indices.append(j)
            current_rank = new_rank
            
    basis = A_np[:, basis_indices]
    return basis.tolist()
