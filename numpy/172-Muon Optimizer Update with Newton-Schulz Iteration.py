import numpy as np

def muon_update(theta: np.ndarray, grad: np.ndarray, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    # Update B
    B_new = mu * B_prev + grad
    
    # Pre-condition for Newton-Schulz using RMS or Frobenius scaling to keep spectral norm < 1
    # Often standard Muon uses a scaling block
    scale = np.sum(B_new ** 2) ** 0.5
    X = B_new / (scale + 1e-8)
    
    # 5 iterations of Newton-Schulz
    for _ in range(5):
        A = X.T @ X
        X = 0.5 * X @ (3.0 * np.eye(X.shape[1]) - A)
        
    O = X
    theta_new = theta - lr * O
    
    return theta_new, B_new, O
