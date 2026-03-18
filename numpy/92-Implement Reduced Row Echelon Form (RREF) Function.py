import numpy as np

def rref(matrix):
    """
    Convert a given matrix into its Reduced Row Echelon Form (RREF).
    """
    A = np.array(matrix, dtype=float)
    if A.size == 0:
        return A
        
    rows, cols = A.shape
    r = 0
    
    for c in range(cols):
        if r >= rows:
            break
            
        # Find pivot in current column
        pivot_row = r
        while pivot_row < rows and np.isclose(A[pivot_row, c], 0):
            pivot_row += 1
            
        if pivot_row == rows:
            continue # No pivot in this column
            
        # Swap rows to bring pivot to current row
        if pivot_row != r:
            A[[r, pivot_row]] = A[[pivot_row, r]]
            
        # Normalize pivot row
        pivot_val = A[r, c]
        A[r] = A[r] / pivot_val
        
        # Eliminate other entries in current column
        for i in range(rows):
            if i != r:
                factor = A[i, c]
                A[i] = A[i] - factor * A[r]
                
        r += 1
        
    # Clean up very small numbers caused by floating point inaccuracies
    A[np.isclose(A, 0)] = 0.0
    return A
