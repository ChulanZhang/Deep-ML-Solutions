"""Deep-ML #320. https://www.deep-ml.com/problems/320

"""
import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if p.numel() == 0 or torch.any((p <= 0) | (p > 1)):
        raise ValueError('Token probabilities must be nonempty and in (0, 1].')
    return torch.exp(-p.log().mean()).item()
