import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores, dtype=float)
    max_score = np.max(scores)
    # Using the log-sum-exp trick for numerical stability
    lse = max_score + np.log(np.sum(np.exp(scores - max_score)))
    return np.round(scores - lse, 4)
