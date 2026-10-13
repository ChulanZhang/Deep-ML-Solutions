def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """
    Calculate the covariance matrix for a given set of vectors.
    Each inner list represents a feature with its observations.
    
    Args:
        vectors: List of features, where each feature is a list of observations.
    
    Returns:
        Covariance matrix as a list of lists.
    """
    n_features = len(vectors)
    if n_features == 0:
        return []
    
    n_observations = len(vectors[0])
    if n_observations <= 1:
        # Covariance requires at least 2 observations for sample covariance
        return [[0.0] * n_features for _ in range(n_features)]
        
    # 1. Calculate means for each feature
    means = [sum(feature) / n_observations for feature in vectors]
    
    # 2. Initialize the covariance matrix (n_features x n_features)
    cov_matrix = [[0.0] * n_features for _ in range(n_features)]
    
    # 3. Calculate covariance for each pair of features
    # Formula: cov(X, Y) = sum((x_i - mean_x) * (y_i - mean_y)) / (n_observations - 1)
    for i in range(n_features):
        for j in range(n_features): # Can be optimized since matrix is symmetric, but explicit is fine
            cov_sum = 0.0
            for k in range(n_observations):
                cov_sum += (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
            # Sample covariance divides by (n - 1)
            cov_matrix[i][j] = cov_sum / (n_observations - 1)
            
    return cov_matrix
