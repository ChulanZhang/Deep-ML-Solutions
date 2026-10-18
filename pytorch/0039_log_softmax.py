"""Deep-ML #39: Implementation of Log Softmax Function.
https://www.deep-ml.com/problems/39
Standalone PyTorch solution; public-example replay is not an official submission.
Rounding is an exercise output convention; do not round a training loss.
"""
import torch

def log_softmax(scores: list[float]) -> torch.Tensor:
    x = torch.as_tensor(scores, dtype=torch.float64)
    return torch.round(x - torch.logsumexp(x, dim=-1), decimals=4)
