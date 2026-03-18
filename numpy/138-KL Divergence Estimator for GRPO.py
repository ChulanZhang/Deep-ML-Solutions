import numpy as np

def kl_divergence_estimator(rhos, pi_theta, pi_ref) -> np.ndarray:
    """
    Compute the unbiased KL divergence estimator for GRPO.
    D_KL = rho * (pi_ref/pi_theta - log(pi_ref/pi_theta) - 1)
    """
    rhos_arr = np.array(rhos, dtype=float)
    pt = np.array(pi_theta, dtype=float)
    pr = np.array(pi_ref, dtype=float)
    
    ratio = pr / pt
    kl = rhos_arr * (ratio - np.log(ratio) - 1.0)
    
    return kl
