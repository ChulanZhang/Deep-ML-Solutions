import torch

def feast_predictor(scores: list) -> list:
    scores = torch.tensor(scores, dtype=float)
    max_s = torch.amax(scores)
    exp_scores = torch.exp(scores - max_s)
    probs = exp_scores / torch.sum(exp_scores)
    return torch.round(probs, 4).tolist()
