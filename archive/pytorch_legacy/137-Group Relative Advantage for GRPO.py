import torch

def compute_group_relative_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute the Group Relative Advantage for GRPO.
    """
    R = torch.tensor(rewards, dtype=float)
    mean_R = torch.mean(R)
    std_R = torch.std(R)
    
    if std_R == 0:
        return [0.0] * len(rewards)
        
    A = (R - mean_R) / std_R
    return A.tolist()
