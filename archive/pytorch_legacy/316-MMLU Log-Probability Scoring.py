def mmlu_scoring(logprobs_dict: dict) -> str:
    # logprobs_dict example: {'A': -0.1, 'B': -2.3, 'C': -5.0, 'D': -1.2}
    best_ans = max(logprobs_dict, key=logprobs_dict.get)
    return best_ans
