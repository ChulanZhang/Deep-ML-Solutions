import numpy as np

def xavier_initialization(shape: tuple) -> np.ndarray:
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    else:
        fan_in = np.prod(shape[1:])
        fan_out = shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)
