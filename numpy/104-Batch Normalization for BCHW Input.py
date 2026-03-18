import numpy as np

def batch_normalization(X, gamma, beta, running_mean=None, running_var=None, momentum=0.1, epsilon=1e-5, training=True):
    """
    Implement Batch Normalization for a 4D tensor in BCHW format.
    """
    X = np.array(X, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    
    N, C, H, W = X.shape
    
    if running_mean is None:
        running_mean = np.zeros(C, dtype=float)
    else:
        running_mean = np.array(running_mean, dtype=float)
        
    if running_var is None:
        running_var = np.ones(C, dtype=float)
    else:
        running_var = np.array(running_var, dtype=float)
        
    if training:
        batch_mean = np.mean(X, axis=(0, 2, 3))
        batch_var = np.var(X, axis=(0, 2, 3))
        
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
    X_norm = (X - mean_reshaped) / np.sqrt(var_reshaped + epsilon)
    out = gamma_reshaped * X_norm + beta_reshaped
    
    return out, running_mean, running_var
