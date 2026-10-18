import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.

    Formula: P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
    where C(n, k) is the combinations formula n! / (k! * (n - k)!)

    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial

    Returns:
        Probability of exactly k successes, rounded to 5 decimal places.
    """
    # 1. 计算组合数 C(n, k) 也就是 n 选 k
    # math.comb 在 Python 3.8+ 可用，非常方便；如果是旧版可以用 factorial 组合
    combinations = math.comb(n, k)
    
    # 2. 计算成功的概率部分: p 的 k 次方
    success_prob = p ** k
    
    # 3. 计算失败的概率部分: (1 - p) 的剩余次数 (n - k) 次方
    failure_prob = (1 - p) ** (n - k)
    
    # 4. 组合计算最终概率
    val = combinations * success_prob * failure_prob
    
    # Deep-ML 统计类题目通常期望保留 5 位小数
    return round(val, 5)
