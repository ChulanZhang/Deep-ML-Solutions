import numpy as np

def compute_group_relative_advantage(rewards: list[float]) -> list[float]:
    """
    Compute the Group Relative Advantage for GRPO.
    """
    R = np.array(rewards, dtype=float)
    mean_R = np.mean(R)
    std_R = np.std(R)
    
    if std_R == 0:
        return [0.0] * len(rewards)
        
    A = (R - mean_R) / std_R
    return A.tolist()
