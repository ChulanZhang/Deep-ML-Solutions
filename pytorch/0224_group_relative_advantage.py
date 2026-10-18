"""Deep-ML #224: Group Relative Advantage for GRPO.
https://www.deep-ml.com/problems/224
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def compute_group_relative_advantage(rewards):
    x = rewards if rewards.is_floating_point() else rewards.float()
    centered = x - x.mean()
    std = centered.square().mean().sqrt()
    return centered / std.clamp_min(1e-8)
