import numpy as np

def sgtm_update(param, grad, momentum_buffer, lr=0.01, momentum=0.9):
    # Stand-in structural SGD momentum
    v = momentum * np.array(momentum_buffer) + np.array(grad)
    p = np.array(param) - lr * v
    if np.isscalar(param) or (isinstance(param, np.ndarray) and param.ndim == 0):
        return float(p), float(v)
    return p, v
