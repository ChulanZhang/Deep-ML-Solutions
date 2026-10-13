import torch

def transform_basis(B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Compute transformation matrix P from basis B to basis C.
    P = C^-1 * B
    """
    B_np = torch.tensor(B, dtype=float)
    C_np = torch.tensor(C, dtype=float)
    
    # P = C_inv @ B
    try:
        C_inv = torch.linalg.inv(C_np)
        P = C_inv @ B_np
        P_rounded = torch.round(P, 4)
        return P_rounded.tolist()
    except np.linalg.LinAlgError:
        # Fallback if C is singular, but problem assumes it's a basis
        return []
