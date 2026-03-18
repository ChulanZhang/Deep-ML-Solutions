import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm mathematically.
    """
    N, d = Q.shape
    O = np.zeros((N, d))
    l = np.zeros((N, 1))
    m = np.full((N, 1), -np.inf)
    
    T_r = int(np.ceil(N / block_size))
    T_c = int(np.ceil(N / block_size))
    
    for j in range(T_c):
        start_j = j * block_size
        end_j = min(start_j + block_size, N)
        K_j = K[start_j:end_j, :]
        V_j = V[start_j:end_j, :]
        
        for i in range(T_r):
            start_i = i * block_size
            end_i = min(start_i + block_size, N)
            
            Q_i = Q[start_i:end_i, :]
            O_i = O[start_i:end_i, :]
            l_i = l[start_i:end_i, :]
            m_i = m[start_i:end_i, :]
            
            S_ij = np.dot(Q_i, K_j.T) / np.sqrt(d)
            
            m_ij = np.max(S_ij, axis=1, keepdims=True)
            m_new = np.maximum(m_i, m_ij)
            
            P_ij = np.exp(S_ij - m_new)
            
            l_new = np.exp(m_i - m_new) * l_i + np.sum(P_ij, axis=1, keepdims=True)
            
            O_new = (np.exp(m_i - m_new) * l_i * O_i + np.dot(P_ij, V_j)) / l_new
            
            O[start_i:end_i, :] = O_new
            l[start_i:end_i, :] = l_new
            m[start_i:end_i, :] = m_new
            
    return O
