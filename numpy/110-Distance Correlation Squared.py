import numpy as np

def distance_correlation_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute squared distance correlation (dcor^2) between X and Y.
    """
    N = X.shape[0] # assuming X and Y have shape (N, D1) and (N, D2)
    if N == 0:
        return 0.0
        
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    def pairwise_dist(A):
        A_sq = np.sum(A**2, axis=-1, keepdims=True)
        D_sq = A_sq + A_sq.T - 2 * (A @ A.T)
        D_sq = np.maximum(D_sq, 0.0)
        return np.sqrt(D_sq)
        
    A = pairwise_dist(X)
    B = pairwise_dist(Y)
    
    def double_center(D):
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand_mean = D.mean()
        return D - row_mean - col_mean + grand_mean
        
    A_cent = double_center(A)
    B_cent = double_center(B)
    
    dCov2 = np.sum(A_cent * B_cent) / (N**2)
    dVarX2 = np.sum(A_cent * A_cent) / (N**2)
    dVarY2 = np.sum(B_cent * B_cent) / (N**2)
    
    if dVarX2 <= 0.0 or dVarY2 <= 0.0:
        return 0.0
        
    return float(dCov2 / np.sqrt(dVarX2 * dVarY2))
