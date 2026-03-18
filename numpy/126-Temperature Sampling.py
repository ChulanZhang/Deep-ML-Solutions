import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
    """
    Implement temperature sampling over logits.
    """
    logits_array = np.array(logits, dtype=float)
    if temperature <= 0.0:
        out = np.zeros_like(logits_array)
        out[np.argmax(logits_array)] = 1.0
        return out.tolist()
        
    scaled = logits_array / temperature
    max_val = np.max(scaled)
    exps = np.exp(scaled - max_val)
    probs = exps / np.sum(exps)
    return probs.tolist()
