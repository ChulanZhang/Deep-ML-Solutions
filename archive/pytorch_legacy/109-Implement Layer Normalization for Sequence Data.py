import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization on a (batch_size, seq_len, d_model) tensor.
    """
    mean = torch.mean(X, dim=-1, keepdim=True)
    var = torch.var(X, dim=-1, keepdim=True)
    
    X_norm = (X - mean) / torch.sqrt(var + epsilon)
    
    return gamma * X_norm + beta
