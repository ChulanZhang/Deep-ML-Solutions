import torch

def calculate_perplexity(probabilities: torch.Tensor) -> float:
    """
    Calculate the perplexity of a language model.
    """
    probs = torch.tensor(probabilities, dtype=float)
    n = len(probs)
    nll = -torch.mean(torch.log(probs))
    ppl = torch.exp(nll)
    return round(float(ppl), 4)
