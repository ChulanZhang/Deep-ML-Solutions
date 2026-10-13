def rubric_judge(response: str, rubric: list) -> int:
    score = 0
    for criterion in rubric:
        if criterion.lower() in response.lower():
            score += 1
    return score
