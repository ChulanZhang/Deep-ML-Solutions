"""Deep-ML #49: Implement Adam Optimization Algorithm.
https://www.deep-ml.com/problems/49
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9,
                   beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = torch.as_tensor(x0).clone()
    if not x.is_floating_point():
        x = x.to(torch.float64)
    m, v = torch.zeros_like(x), torch.zeros_like(x)
    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g.square()
        m_hat, v_hat = m / (1 - beta1**t), v / (1 - beta2**t)
        x = x - learning_rate * m_hat / (v_hat.sqrt() + epsilon)
    return x
