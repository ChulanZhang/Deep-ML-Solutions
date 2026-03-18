def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    k = len(rewards)
    val = ((1.0 - alpha) ** k) * Q1
    for i in range(1, k + 1):
        val += alpha * ((1.0 - alpha) ** (k - i)) * rewards[i-1]
    
    # Returning rounded to 5 decimal places to avoid precision artifacts, example is 4.73
    return float(np.round(val, 5)) if 'np' in globals() else round(val, 5)
