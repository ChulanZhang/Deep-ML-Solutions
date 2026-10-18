"""Deep-ML #101. https://www.deep-ml.com/problems/101
This exercise includes rho in the KL term and returns an objective to maximize, not a loss.
"""
import torch

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01):
    r, a, old, ref = (torch.as_tensor(x, dtype=torch.float64)
                      for x in (rhos, A, pi_theta_old, pi_theta_ref))
    if torch.any(r <= 0) or torch.any(old <= 0) or torch.any(ref <= 0):
        raise ValueError('Ratios and likelihoods must be positive.')
    surrogate = torch.minimum(r * a, r.clamp(1 - epsilon, 1 + epsilon) * a)
    log_ratio = ref.log() - (r * old).log()
    kl = r * (torch.expm1(log_ratio) - log_ratio)
    return (surrogate - beta * kl).mean()
