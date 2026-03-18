import numpy as np

def compute_hessian(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    x_arr = np.array(x, dtype=float)
    n = len(x_arr)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x_arr.copy()
            x_pm = x_arr.copy()
            x_mp = x_arr.copy()
            x_mm = x_arr.copy()
            
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h * h)
            
    return np.round(H, 4)
