import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """
    Compute transformation matrix P from basis B to basis C.
    P = C^-1 * B
    """
    B_np = np.array(B, dtype=float)
    C_np = np.array(C, dtype=float)
    
    # P = C_inv @ B
    try:
        C_inv = np.linalg.inv(C_np)
        P = C_inv @ B_np
        P_rounded = np.round(P, 4)
        return P_rounded.tolist()
    except np.linalg.LinAlgError:
        # Fallback if C is singular, but problem assumes it's a basis
        return []
