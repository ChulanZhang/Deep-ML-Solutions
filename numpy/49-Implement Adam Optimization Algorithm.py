import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    """
    Implement Adam optimization algorithm.
    
    Args:
        f: Objective function (not actually needed if we have analytic gradient, but kept for interface)
        grad: Gradient function
        x0: Initial parameters
        learning_rate: alpha
        beta1: decay for 1st moment
        beta2: decay for 2nd moment
        epsilon: stability constant
        num_iterations: total steps
        
    Returns:
        Optimized parameters
    """
    # Initialize moments
    m = np.zeros_like(x0, dtype=np.float64)
    v = np.zeros_like(x0, dtype=np.float64)
    x = np.array(x0, dtype=np.float64)
    
    for t in range(1, num_iterations + 1):
        g = grad(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
    return x
