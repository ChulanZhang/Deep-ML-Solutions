import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the total discounted return for a sequence of rewards.
    """
    G = 0.0
    for i, r in enumerate(rewards):
        G += r * (gamma ** i)
    return float(G)
