"""Deep-ML #429. https://www.deep-ml.com/problems/429

"""
import torch

def quantization_quality_check(log_probs_original, log_probs_quantized, thresholds=None):
    thresholds = {'excellent': 1.0, 'acceptable': 5.0} if thresholds is None else thresholds
    original = torch.as_tensor(log_probs_original, dtype=torch.float64)
    quantized = torch.as_tensor(log_probs_quantized, dtype=torch.float64)
    if original.numel() == 0 or original.shape != quantized.shape:
        raise ValueError('Need equal nonempty token arrays.')
    a, b = -original.mean(), -quantized.mean()
    p, q = a.exp().item(), b.exp().item()
    percent = torch.expm1(b - a).item() * 100
    quality = ('excellent' if percent <= thresholds['excellent'] else
               'acceptable' if percent <= thresholds['acceptable'] else 'poor')
    return {'perplexity_original': round(p, 4), 'perplexity_quantized': round(q, 4),
            'perplexity_delta': round(q - p, 4),
            'relative_delta_percent': round(percent, 4), 'quality': quality}
