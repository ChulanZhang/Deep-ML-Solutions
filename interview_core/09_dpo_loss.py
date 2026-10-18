"""Differentiable DPO loss; related #382 returns a rounded reporting dict.
Inputs are sequence log-probabilities (already summed over response tokens).
Reference log-probabilities should come from a frozen/no-grad reference model.
"""
import torch
from torch.nn import functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen = policy_chosen_logps - ref_chosen_logps
    rejected = policy_rejected_logps - ref_rejected_logps
    return -F.logsigmoid(beta * (chosen - rejected)).mean()
