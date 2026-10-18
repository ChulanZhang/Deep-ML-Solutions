"""Two-layer ReLU MLP, manual gradients for 0.5 * mean((prediction-Y)^2).
The mean covers BOTH batch and output dimensions: divide by prediction.numel().
This is an interview extension, not a claimed one-to-one website problem.
"""
import torch

def mlp_forward_backward(X, Y, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = z1.relu()
    prediction = a1 @ W2 + b2
    error = prediction - Y
    loss = .5 * error.square().mean()
    dz2 = error / prediction.numel()
    dw2, db2 = a1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (z1 > 0).to(z1.dtype)
    dw1, db1 = X.T @ dz1, dz1.sum(0)
    return loss, (dw1, db1, dw2, db2)
