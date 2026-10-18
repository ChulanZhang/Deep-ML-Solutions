import math

def poisson_probability(k: int, lam: float) -> float:
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    
    Formula: P(X = k) = (lam^k * e^-lam) / k!
    
    Args:
        k: Number of events (non-negative integer)
        lam: The average rate (mean) of occurrences in a fixed interval
        
    Returns:
        The probability rounded to 5 decimal places.
    """
    # 1. 计算分子: lambda 的 k 次方 乘以 e 的 -lambda 次方
    # math.exp(x) 计算 e^x
    numerator = (lam ** k) * math.exp(-lam)
    
    # 2. 计算分母: k 的阶乘
    denominator = math.factorial(k)
    
    # 3. 计算概率
    val = numerator / denominator
    
    # 4. 根据题目要求，保留 5 位小数
    return round(val, 5)
