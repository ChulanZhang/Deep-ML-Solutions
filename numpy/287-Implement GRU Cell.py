import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray, W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray, 
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray, 
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
             
    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    
    x = np.array(x, dtype=float)
    h_prev = np.array(h_prev, dtype=float)
    
    z_t = sigmoid(np.dot(x, W_z) + np.dot(h_prev, U_z) + b_z)
    r_t = sigmoid(np.dot(x, W_r) + np.dot(h_prev, U_r) + b_r)
    
    h_tilde = np.tanh(np.dot(x, W_h) + np.dot(r_t * h_prev, U_h) + b_h)
    
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    return np.round(h_t, 4)
