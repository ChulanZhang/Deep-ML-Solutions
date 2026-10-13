import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    """
    Solve Ax = b using the Gauss-Seidel method.
    """
    if x_ini is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = x_ini.copy().astype(float)
        
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    size = len(b)
    
    for _ in range(n):
        for i in range(size):
            s = 0.0
            for j in range(size):
                if j != i:
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]
            
    return x.tolist() if isinstance(x, np.ndarray) else x
