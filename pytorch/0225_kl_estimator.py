"""Deep-ML #225: KL Divergence Estimator for GRPO.
https://www.deep-ml.com/problems/225
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def kl_divergence_estimator(pi_theta, pi_ref):
    if torch.any(pi_theta <= 0) or torch.any(pi_ref <= 0):
        raise ValueError('Probabilities must be positive.')
    log_ratio = torch.log(pi_ref) - torch.log(pi_theta)
    return torch.expm1(log_ratio) - log_ratio
