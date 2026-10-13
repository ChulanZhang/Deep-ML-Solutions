import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization (Batch, Stochastic, or Mini-batch).
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', 'mini_batch')
    
    Returns:
        final_weights: Updated weights after optimization
    """
    m = X.shape[0]
    
    for epoch in range(n_epochs):
        if method == 'batch':
            # Use all data for one update
            predictions = np.dot(X, weights)
            error = predictions - y
            gradient = (1/m) * np.dot(X.T, error)
            weights = weights - learning_rate * gradient
            
        elif method == 'stochastic':
            # Update for each individual sample sequentially
            for i in range(m):
                x_i = X[i].reshape(1, -1) # Shape (1, n)
                y_i = y[i]
                prediction = np.dot(x_i, weights)
                error = prediction - y_i
                # gradient for one sample
                gradient = np.dot(x_i.T, error).reshape(-1)
                weights = weights - learning_rate * gradient
                
        elif method == 'mini_batch':
            # Update for batches of size `batch_size` consecutively
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X[i:end_idx]
                y_batch = y[i:end_idx]
                batch_m = end_idx - i
                
                predictions = np.dot(X_batch, weights)
                error = predictions - y_batch
                gradient = (1/batch_m) * np.dot(X_batch.T, error)
                weights = weights - learning_rate * gradient
                
    return weights
