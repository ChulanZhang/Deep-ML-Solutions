import torch

def solve_jacobi(A: torch.Tensor, b: torch.Tensor, n: int) -> list:
    """
    Use Jacobi method to solve Ax = b.
    Iterates exactly n times.
    Rounds intermediate solutions to 4 decimal places.
    Initializes x to all zeros.
    """
    A = torch.tensor(A, dtype=float)
    b = torch.tensor(b, dtype=float)
    size = len(b)
    
    x = torch.zeros(size, dtype=float)
    
    for _ in range(n):
        x_new = torch.zeros_like(x)
        for i in range(size):
            # sum_{j != i} A[i, j] * x[j]
            s = sum(A[i, j] * x[j] for j in range(size) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
            
        # Round intermediate solutions to 4 decimal places
        x = torch.round(x_new, 4)
        
    return x.tolist()
