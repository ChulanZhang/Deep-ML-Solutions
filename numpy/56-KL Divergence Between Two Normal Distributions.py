import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    """
    Calculate the KL Divergence between two univariate normal distributions.
    
    Formula: D_KL(P || Q) = log(sigma_q / sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2 * sigma_q^2) - 0.5
    """
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    return float(term1 + term2 - 0.5)
