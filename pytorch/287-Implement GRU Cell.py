import torch

def gru_cell(x: torch.Tensor, h_prev: torch.Tensor, W_z: torch.Tensor, U_z: torch.Tensor, b_z: torch.Tensor, 
             W_r: torch.Tensor, U_r: torch.Tensor, b_r: torch.Tensor, 
             W_h: torch.Tensor, U_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
             
    def sigmoid(z): return 1.0 / (1.0 + torch.exp(-z))
    
    x = torch.tensor(x, dtype=float)
    h_prev = torch.tensor(h_prev, dtype=float)
    
    z_t = sigmoid(torch.matmul(x, W_z) + torch.matmul(h_prev, U_z) + b_z)
    r_t = sigmoid(torch.matmul(x, W_r) + torch.matmul(h_prev, U_r) + b_r)
    
    h_tilde = np.tanh(torch.matmul(x, W_h) + torch.matmul(r_t * h_prev, U_h) + b_h)
    
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    return torch.round(h_t, 4)
