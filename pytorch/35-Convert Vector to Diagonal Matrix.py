import torch

def make_diagonal(x):
    """
    Convert a 1D numpy array into a diagonal matrix.
    """
    x = torch.tensor(x, dtype=float)
    n = len(x)
    diag_matrix = torch.zeros((n, n), dtype=float)
    for i in range(n):
        diag_matrix[i, i] = x[i]
    return diag_matrix.tolist()
