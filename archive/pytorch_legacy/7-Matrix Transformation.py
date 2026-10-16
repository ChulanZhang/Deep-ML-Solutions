import torch

def transform_matrix(A: torch.Tensor, T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    Transforms matrix A using operation T^-1 * A * S
    """
    A_np = torch.tensor(A, dtype=float)
    T_np = torch.tensor(T, dtype=float)
    S_np = torch.tensor(S, dtype=float)
    
    try:
        # Check determinants
        det_T = torch.linalg.det(T_np)
        det_S = torch.linalg.det(S_np)
        
        if np.isclose(det_T, 0) or np.isclose(det_S, 0):
            return -1
            
        T_inv = torch.linalg.inv(T_np)
        res = T_inv @ A_np @ S_np
        return res.tolist()
    except np.linalg.LinAlgError:
        return -1
