import torch

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm mathematically.
    """
    N, d = Q.shape
    O = torch.zeros((N, d))
    l = torch.zeros((N, 1))
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
            
            S_ij = torch.matmul(Q_i, K_j.T) / torch.sqrt(d)
            
            m_ij = torch.amax(S_ij, dim=1, keepdim=True)
            m_new = torch.maximum(m_i, m_ij)
            
            P_ij = torch.exp(S_ij - m_new)
            
            l_new = torch.exp(m_i - m_new) * l_i + torch.sum(P_ij, dim=1, keepdim=True)
            
            O_new = (torch.exp(m_i - m_new) * l_i * O_i + torch.matmul(P_ij, V_j)) / l_new
            
            O[start_i:end_i, :] = O_new
            l[start_i:end_i, :] = l_new
            m[start_i:end_i, :] = m_new
            
    return O
