import torch

def mean_ablation(activations: torch.Tensor, indices: list) -> torch.Tensor:
    activations = torch.tensor(activations, dtype=float)
    mean_val = torch.mean(activations)
    for idx in indices:
        activations[idx] = mean_val
    return torch.round(activations, 4)
