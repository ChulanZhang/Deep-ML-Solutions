import torch

def pca(data: torch.Tensor, k: int) -> torch.Tensor:
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # 1. Standardize the data
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    std[std == 0] = 1.0  # Avoid division by zero
    standardized_data = (data - mean) / std
    
    # 2. Compute the covariance matrix
    # rowvar=False treats columns as features
    cov_matrix = np.cov(standardized_data, rowvar=False)
    
    # 3. Compute eigenvalues and eigenvectors
    # using eigh because covariance matrix is symmetric
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # 4. Sort eigen pairs in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Extract top k components
    pcs = eigenvectors[:, :k]
    
    # 6. Apply sign convention to ensure deterministic output
    for i in range(k):
        for j in range(pcs.shape[0]):
            if torch.abs(pcs[j, i]) > 1e-10:
                if pcs[j, i] < 0:
                    pcs[:, i] *= -1
                break
                
    return torch.round(pcs, 4)
