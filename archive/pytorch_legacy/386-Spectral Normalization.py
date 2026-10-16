import torch

def spectral_normalization(W: torch.Tensor, u: torch.Tensor, n_power_iterations: int = 1) -> tuple:
    W_mat = torch.tensor(W, dtype=float).reshape(W.shape[0], -1)
    u_vec = torch.tensor(u, dtype=float)
    
    for _ in range(n_power_iterations):
        v = torch.matmul(W_mat.T, u_vec)
        v = v / (np.linalg.norm(v) + 1e-12)
        
        u_vec = torch.matmul(W_mat, v)
        u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-12)
        
    sigma = torch.matmul(u_vec.T, torch.matmul(W_mat, v))
    W_sn = W_mat / sigma
    
    return torch.round(W_sn.reshape(W.shape), 4), torch.round(u_vec, 4)
