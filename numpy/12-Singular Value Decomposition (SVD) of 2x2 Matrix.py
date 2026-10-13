import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation to diagonalize A^T A.
    """
    A = np.array(A, dtype=float)
    
    # 1. Compute A^T A
    ATA = A.T @ A
    a = ATA[0, 0]
    b = ATA[1, 1]
    c = ATA[0, 1]  # == ATA[1, 0]
    
    # 2. Find Jacobi rotation angle theta
    # We want to eliminate 'c'. tan(2*theta) = 2c / (a - b)
    if np.isclose(c, 0):
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2 * c, a - b)
        
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Right singular vectors (V)
    V = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # Check if we need to swap to sort singular values in descending order
    # (cos_t, sin_t) diagonalizes ATA into S^2.
    # Diagonal elements of V^T * ATA * V
    S2 = np.diag(V.T @ ATA @ V)
    
    if S2[1] > S2[0]:
        # Swap columns of V
        V = V[:, [1, 0]]
        S2 = S2[[1, 0]]
        
    # Singular values
    S = np.sqrt(np.maximum(S2, 0)) # prevent negative near zero
    
    # Left singular vectors (U)
    # AV = U Sigma => U_i = A V_i / S_i
    U = np.zeros((2, 2))
    for i in range(2):
        if S[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / S[i]
        else:
            # If S[i] is 0, make it orthogonal to the other
            if i == 1:
                U[:, 1] = np.array([-U[1, 0], U[0, 0]])
            else:
                U[:, 0] = np.array([1.0, 0.0]) # fallback
                
    return (U, S, V.T)
