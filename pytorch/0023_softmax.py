"""Deep-ML #23: Softmax Activation Function Implementation.
https://www.deep-ml.com/problems/23
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def softmax(scores: list[float]) -> list[float]:
    x = torch.as_tensor(scores, dtype=torch.float64)
    if x.numel() == 0:
        return []
    values = torch.exp(x - x.max())
    return (values / values.sum()).tolist()
