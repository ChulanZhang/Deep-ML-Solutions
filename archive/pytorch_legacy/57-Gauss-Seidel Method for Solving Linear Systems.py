import torch

def gauss_seidel(A, b, n, x_ini=None):
    """
    Solve Ax = b using the Gauss-Seidel method.
    """
    if x_ini is None:
        x = torch.zeros_like(b, dtype=float)
    else:
        x = x_ini.copy().to(torch.float32)
        
    A = torch.tensor(A, dtype=float)
    b = torch.tensor(b, dtype=float)
    size = len(b)
    
    for _ in range(n):
        for i in range(size):
            s = 0.0
            for j in range(size):
                if j != i:
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]
            
    return x.tolist() if isinstance(x, torch.Tensor) else x
