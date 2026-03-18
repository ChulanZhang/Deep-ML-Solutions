import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.p > 0.0:
            # We scale during training to maintain expected value
            self.mask = (np.random.rand(*x.shape) >= self.p).astype(float)
            return x * self.mask / (1.0 - self.p)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.mask is not None and self.p > 0.0:
            return grad * self.mask / (1.0 - self.p)
        return grad
