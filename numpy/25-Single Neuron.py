import math

def sigmoid(z: float) -> float:
    # Handle overflow possibility gracefully
    if z < -100:
        return 0.0
    if z > 100:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with a sigmoid activation function for binary classification.
    
    Args:
        features: Multidimensional input features (m samples, n features)
        labels: True binary labels (m samples)
        weights: One for each feature (n weights)
        bias: Bias term
        
    Returns:
        (probabilities, mse): predicted probabilities and Mean Squared Error rounded to 4 decimals.
    """
    m = len(features)
    if m == 0:
        return [], 0.0
        
    probabilities = []
    mse_sum = 0.0
    
    for i in range(m):
        # 1. Linear combination: z = w·x + b
        # Using zip to multiply corresponding elements of a feature row and weights
        z = sum(x * w for x, w in zip(features[i], weights)) + bias
        
        # 2. Activation: a = sigmoid(z)
        prob = sigmoid(z)
        rounded_prob = round(prob, 4)
        probabilities.append(rounded_prob)
        
        # 3. Error accumulation for MSE
        error = rounded_prob - labels[i]
        mse_sum += error ** 2
        
    # Calculate MSE and round
    mse = round(mse_sum / m, 4)
    
    return probabilities, mse
