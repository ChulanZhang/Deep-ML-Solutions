import torch

def paris_expert_consensus(expert_predictions: list) -> torch.Tensor:
    preds = torch.tensor(expert_predictions, dtype=float)
    consensus = torch.mean(preds, dim=0)
    return torch.round(consensus, 4)
