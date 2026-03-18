import numpy as np

def transform_matrix(A: list[list[float]], T: list[list[float]], S: list[list[float]]) -> list[list[float]]:
    """
    Transforms matrix A using operation T^-1 * A * S
    """
    A_np = np.array(A, dtype=float)
    T_np = np.array(T, dtype=float)
    S_np = np.array(S, dtype=float)
    
    try:
        # Check determinants
        det_T = np.linalg.det(T_np)
        det_S = np.linalg.det(S_np)
        
        if np.isclose(det_T, 0) or np.isclose(det_S, 0):
            return -1
            
        T_inv = np.linalg.inv(T_np)
        res = T_inv @ A_np @ S_np
        return res.tolist()
    except np.linalg.LinAlgError:
        return -1
