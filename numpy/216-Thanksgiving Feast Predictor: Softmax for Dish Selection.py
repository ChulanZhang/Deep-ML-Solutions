import numpy as np

def feast_predictor(scores: list) -> list:
    scores = np.array(scores, dtype=float)
    max_s = np.max(scores)
    exp_scores = np.exp(scores - max_s)
    probs = exp_scores / np.sum(exp_scores)
    return np.round(probs, 4).tolist()
