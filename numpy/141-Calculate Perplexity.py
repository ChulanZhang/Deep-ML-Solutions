import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model.
    """
    probs = np.array(probabilities, dtype=float)
    n = len(probs)
    nll = -np.mean(np.log(probs))
    ppl = np.exp(nll)
    return round(float(ppl), 4)
