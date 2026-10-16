import torch

def kl_divergence_estimator(rhos, pi_theta, pi_ref) -> torch.Tensor:
    """
    Compute the unbiased KL divergence estimator for GRPO.
    D_KL = rho * (pi_ref/pi_theta - log(pi_ref/pi_theta) - 1)
    """
    rhos_arr = torch.tensor(rhos, dtype=float)
    pt = torch.tensor(pi_theta, dtype=float)
    pr = torch.tensor(pi_ref, dtype=float)
    
    ratio = pr / pt
    kl = rhos_arr * (ratio - torch.log(ratio) - 1.0)
    
    return kl
