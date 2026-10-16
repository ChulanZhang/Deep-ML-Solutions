import torch

class DropoutLayer:
    def __init__(self, p: float):
        self.p = p
        self.mask = None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Generate mask during training, store in self.mask, and return scaled x."""
        if training:
            # probability of keeping is 1 - p
            self.mask = (torch.rand(*x.shape) >= self.p).to(torch.float32)
            return x * self.mask / (1.0 - self.p)
        else:
            return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply stored self.mask to grad and return."""
        return grad * self.mask / (1.0 - self.p)
