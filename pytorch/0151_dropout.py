"""Deep-ML #151: Dropout Layer.
https://www.deep-ml.com/problems/151
Standalone PyTorch solution; public-example replay is not an official submission.
Uses the PyTorch RNG, not NumPy RNG. Eval leaves the last training mask untouched.
"""
import torch

class DropoutLayer:
    def __init__(self, p: float):
        if not 0 <= p < 1:
            raise ValueError('p must be in [0, 1).')
        self.p, self.mask = p, None

    def forward(self, x, training=True):
        if not training:
            return x
        self.mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        return x * self.mask / (1 - self.p)

    def backward(self, grad):
        if self.mask is None:
            raise ValueError('A training forward pass must precede backward.')
        return grad * self.mask / (1 - self.p)
