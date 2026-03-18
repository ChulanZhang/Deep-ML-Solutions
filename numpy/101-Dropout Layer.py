import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate mask during training, store in self.mask, and return scaled x."""
        if training:
            # probability of keeping is 1 - p
            self.mask = (np.random.rand(*x.shape) >= self.p).astype(float)
            return x * self.mask / (1.0 - self.p)
        else:
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Apply stored self.mask to grad and return."""
        return grad * self.mask / (1.0 - self.p)
