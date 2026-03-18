import numpy as np

def grpo_objective(rhos: list, A: list, pi_theta_old: list, pi_theta_ref: list, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function wrapper for scalar arrays.
    """
    rhos = np.array(rhos)
    A = np.array(A)
    pi_old = np.array(pi_theta_old)
    pi_ref = np.array(pi_theta_ref)
    
    # 1. Clipped Surrogate Objective
    unclipped = rhos * A
    clipped = np.clip(rhos, 1.0 - epsilon, 1.0 + epsilon) * A
    surrogate = np.minimum(unclipped, clipped)
    
    # 2. KL Penalty
    kl = (pi_old / pi_ref) - np.log(pi_old / pi_ref) - 1.0
    
    objective = surrogate - beta * kl
    return float(np.mean(objective))
