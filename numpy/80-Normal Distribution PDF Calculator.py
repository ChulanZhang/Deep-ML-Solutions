import math

def normal_pdf(x: float, mean: float, std_dev: float) -> float:
    """
    Calculate the probability density function (PDF) of the normal distribution.
    
    Formula: PDF(x) = (1 / (std_dev * sqrt(2 * pi))) * e^(-0.5 * ((x - mean) / std_dev)^2)
    
    Args:
        x: The value at which the PDF is evaluated.
        mean: The mean (μ) of the distribution.
        std_dev: The standard deviation (σ) of the distribution.
        
    Returns:
        The evaluated PDF value, rounded to 5 decimal places.
    """
    # 1. 计算公式左半部分: 1 / (σ * √(2π))
    # math.sqrt 是开平方，math.pi 是圆周率
    coefficient = 1.0 / (std_dev * math.sqrt(2 * math.pi))
    
    # 2. 计算指数部分: e 的负多少次方
    exponent = -0.5 * (((x - mean) / std_dev) ** 2)
    
    # 3. 计算完整公式的值
    val = coefficient * math.exp(exponent)
    
    # 4. 根据 Deep-ML 要求保留 5 位小数
    return round(val, 5)
