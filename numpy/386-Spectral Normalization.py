import numpy as np

def spectral_normalization(W: np.ndarray, u: np.ndarray, n_power_iterations: int = 1) -> tuple:
    W_mat = np.array(W, dtype=float).reshape(W.shape[0], -1)
    u_vec = np.array(u, dtype=float)
    
    for _ in range(n_power_iterations):
        v = np.dot(W_mat.T, u_vec)
        v = v / (np.linalg.norm(v) + 1e-12)
        
        u_vec = np.dot(W_mat, v)
        u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-12)
        
    sigma = np.dot(u_vec.T, np.dot(W_mat, v))
    W_sn = W_mat / sigma
    
    return np.round(W_sn.reshape(W.shape), 4), np.round(u_vec, 4)
