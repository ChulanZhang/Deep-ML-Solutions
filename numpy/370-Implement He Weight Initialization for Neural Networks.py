import numpy as np

def he_initialization(shape: tuple) -> np.ndarray:
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:])
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std
