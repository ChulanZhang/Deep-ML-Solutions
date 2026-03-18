import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float, learning_rate: float, epochs: int):
    """
    Simulate a single neuron training process with backpropagation for binary classification.
    
    Args:
        features: numpy array of shape (m, n)
        labels: numpy array of shape (m,)
        weights: numpy array of shape (n,)
        bias: float
        learning_rate: float
        epochs: int
        
    Returns:
        updated_weights (list or array), updated_bias (float), mse_losses (list of floats)
    """
    mse_losses = []
    m = features.shape[0]
    
    for epoch in range(epochs):
        # --- FORWARD PASS ---
        # 1. Calculate linear combination: z = Xw + b
        z = np.dot(features, weights) + bias
        
        # 2. Apply sigmoid activation
        probabilities = 1.0 / (1.0 + np.exp(-z))
        
        # 3. Calculate Mean Squared Error (MSE) loss
        error = probabilities - labels
        mse = np.sum(error ** 2) / m
        mse_losses.append(round(mse, 4))
        
        # --- BACKWARD PASS (Backpropagation) ---
        # Derivative of MSE with respect to predictions (probabilities)
        # dL/da = (2/m) * (a - y)
        dL_da = (2.0 / m) * error
        
        # Derivative of sigmoid activation with respect to its input z
        # da/dz = a * (1 - a)
        da_dz = probabilities * (1.0 - probabilities)
        
        # Chain rule to get derivative of Loss with respect to z
        # dL/dz = dL/da * da/dz
        dL_dz = dL_da * da_dz
        
        # Gradients for weights and bias
        # dL/dw = (dL/dz) * (dz/dw) -> dz/dw = X
        # dL/db = (dL/dz) * (dz/db) -> dz/db = 1
        dL_dw = np.dot(features.T, dL_dz)
        dL_db = np.sum(dL_dz)
        
        # --- UPDATE WEIGHTS AND BIAS ---
        # Gradient Descent Step
        weights = weights - learning_rate * dL_dw
        bias = bias - learning_rate * dL_db
        
    # Return formatted outputs (often expected as list for weights, rounded to 4 decimals)
    return np.round(weights, 4).tolist(), round(bias, 4), mse_losses
