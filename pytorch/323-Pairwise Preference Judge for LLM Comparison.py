def pairwise_judge(prompt: str, response_a: str, response_b: str, judge_func) -> str:
    score_a = judge_func(prompt, response_a)
    score_b = judge_func(prompt, response_b)
    if score_a > score_b: return "A"
    elif score_b > score_a: return "B"
    return "Tie"
