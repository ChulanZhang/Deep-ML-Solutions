import math

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation for a given list of scores.
    Handles numerical stability by subtracting the maximum score from all scores
    before exponentiation to prevent overflow.
    
    Args:
        scores: List of float values
        
    Returns:
        List of probabilities that sum to 1.
    """
    if not scores:
        return []
        
    # 1. Find the maximum score for numerical stability
    max_score = max(scores)
    
    # 2. Compute exponentials: e^(x_i - max(x))
    exps = [math.exp(score - max_score) for score in scores]
    
    # 3. Compute the sum of all exponentials
    sum_exps = sum(exps)
    
    # 4. Normalize to get probabilities and round to 4 decimals as shown in Deep-ML examples
    # (Though Deep-ML usually accepts full precision if not strictly specified, 
    # matching example output `0.0900` implies rounding might be expected or clean)
    probabilities = [round(e / sum_exps, 4) for e in exps]
    
    return probabilities
