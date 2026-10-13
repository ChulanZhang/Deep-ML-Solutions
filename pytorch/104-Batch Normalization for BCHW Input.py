import torch

def batch_normalization(X, gamma, beta, running_mean=None, running_var=None, momentum=0.1, epsilon=1e-5, training=True):
    """
    Implement Batch Normalization for a 4D tensor in BCHW format.
    """
    X = torch.tensor(X, dtype=float)
    gamma = torch.tensor(gamma, dtype=float)
    beta = torch.tensor(beta, dtype=float)
    
    N, C, H, W = X.shape
    
    if running_mean is None:
        running_mean = torch.zeros(C, dtype=float)
    else:
        running_mean = torch.tensor(running_mean, dtype=float)
        
    if running_var is None:
        running_var = torch.ones(C, dtype=float)
    else:
        running_var = torch.tensor(running_var, dtype=float)
        
    if training:
        batch_mean = torch.mean(X, dim=(0, 2, 3))
        batch_var = torch.var(X, dim=(0, 2, 3))
        
        # Update running stats
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        
        # Calculate unbiased variance for the running stat update
        n_elements = N * H * W
        if n_elements > 1:
            unbiased_var = batch_var * n_elements / (n_elements - 1)
        else:
            unbiased_var = batch_var
            
        running_var = (1 - momentum) * running_var + momentum * unbiased_var
        
        mean_use = batch_mean
        var_use = batch_var
    else:
        mean_use = running_mean
        var_use = running_var
        
    # Reshape for broadcasting (1, C, 1, 1)
    mean_reshaped = mean_use.reshape(1, C, 1, 1)
    var_reshaped = var_use.reshape(1, C, 1, 1)
    gamma_reshaped = gamma.reshape(1, C, 1, 1)
    beta_reshaped = beta.reshape(1, C, 1, 1)
    
    # Normalize
    X_norm = (X - mean_reshaped) / torch.sqrt(var_reshaped + epsilon)
    out = gamma_reshaped * X_norm + beta_reshaped
    
    return out, running_mean, running_var
