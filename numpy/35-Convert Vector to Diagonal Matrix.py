import numpy as np

def make_diagonal(x):
    """
    Convert a 1D numpy array into a diagonal matrix.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    diag_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        diag_matrix[i, i] = x[i]
    return diag_matrix.tolist()
