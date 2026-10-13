import torch

def temperature_sampling(logits: torch.Tensor, temperature: float) -> list:
    """
    Implement temperature sampling over logits.
    """
    logits_array = torch.tensor(logits, dtype=float)
    if temperature <= 0.0:
        out = torch.zeros_like(logits_array)
        out[np.argmax(logits_array)] = 1.0
        return out.tolist()
        
    scaled = logits_array / temperature
    max_val = torch.amax(scaled)
    exps = torch.exp(scaled - max_val)
    probs = exps / torch.sum(exps)
    return probs.tolist()
