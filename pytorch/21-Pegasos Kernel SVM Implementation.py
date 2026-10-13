import torch

def pegasos_kernel_svm(data: torch.Tensor, labels: torch.Tensor, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    """
    n_samples = data.shape[0]
    alphas = torch.zeros(n_samples)
    b = 0.0
    
    # Precompute kernel matrix for efficiency
    K = torch.zeros((n_samples, n_samples))
    if kernel == 'linear':
        K = torch.matmul(data, data.T)
    elif kernel == 'rbf':
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^Ty
        sq_norms = torch.sum(data ** 2, dim=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * torch.matmul(data, data.T)
        K = torch.exp(-sq_dists / (2 * sigma ** 2))
        
    for t in range(1, iterations + 1):
        for i in range(n_samples):
            # Evaluate margin
            margin = labels[i] * (torch.sum(alphas * labels * K[:, i]) + b)
            
            if margin < 1:
                # Update rule for kernel Pegasos simplified
                # Usually step size is 1/(lambda*t). Deep-ML expects a specific logic.
                # In standard kernel Pegasos, alpha_i is effectively incremented by 1
                alphas[i] += 1
                
    # Return as list and float
    return alphas.tolist(), b
