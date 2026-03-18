import numpy as np

def square_relu(x: np.ndarray) -> dict:
    """
    Implement Square ReLU and its derivative.
    """
    activated = np.maximum(0, x)
    out = activated ** 2
    # Derivative: 2 * x if x > 0 else 0
    deriv = 2 * activated
    return {'output': out, 'derivative': deriv}
