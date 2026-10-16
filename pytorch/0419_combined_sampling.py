"""Deep-ML #419: Combined Token Sampling Pipeline (Temperature + Top-k + Top-p).
https://www.deep-ml.com/problems/419
Standalone PyTorch solution; public-example replay is not an official submission.
Uses a local PyTorch RNG (does not reset the caller's global RNG).
"""
import torch

def combined_sampling(logits, temperature=1.0, top_k=0, top_p=1.0, seed=42):
    x = torch.as_tensor(logits, dtype=torch.float64)
    if x.ndim != 1 or x.numel() == 0:
        raise ValueError('Expected a nonempty 1D vocabulary.')
    if temperature <= 0:
        index = int(x.argmax())
        probs = torch.zeros_like(x); probs[index] = 1
        return {'probabilities': probs.tolist(), 'sampled_token': index}
    if not 0 < top_p <= 1 or top_k < 0:
        raise ValueError('top_p must be in (0, 1], top_k must be non-negative.')
    x = x / temperature
    if 0 < top_k < x.numel():
        order = torch.argsort(x, descending=True, stable=True)[:top_k]
        mask = torch.ones_like(x, dtype=torch.bool); mask[order] = False
        x = x.masked_fill(mask, float('-inf'))
    probs = torch.softmax(x, dim=-1)
    order = torch.argsort(probs, descending=True, stable=True)
    ordered = probs[order]
    keep = (ordered.cumsum(-1) - ordered) < top_p
    kept = ordered * keep
    probs = torch.zeros_like(probs).scatter(0, order, kept / kept.sum())
    generator = torch.Generator(device='cpu').manual_seed(seed)
    index = int(torch.multinomial(probs, 1, generator=generator))
    return {'probabilities': [round(v, 4) for v in probs.tolist()], 'sampled_token': index}
