import numpy as np

def dpo_loss(pi_theta_w: float, pi_ref_w: float, pi_theta_l: float, pi_ref_l: float, beta: float) -> float:
    log_ratio_w = pi_theta_w - pi_ref_w
    log_ratio_l = pi_theta_l - pi_ref_l
    
    logits = beta * (log_ratio_w - log_ratio_l)
    def sigmoid(x): return 1.0 / (np.exp(-x) + 1.0)
    
    loss = -np.log(np.clip(sigmoid(logits), 1e-15, 1.0))
    return float(np.round(loss, 4))
