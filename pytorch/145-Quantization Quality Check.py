import torch

def quantization_quality_check(log_probs_original: list, log_probs_quantized: list, thresholds: dict = None) -> dict:
    """
    Evaluate quantization quality by comparing perplexity before and after quantization.
    """
    if thresholds is None:
        thresholds = {'excellent': 1.0, 'acceptable': 5.0}
        
    orig = torch.tensor(log_probs_original)
    quant = torch.tensor(log_probs_quantized)
    
    ppl_orig = torch.exp(-torch.mean(orig))
    ppl_quant = torch.exp(-torch.mean(quant))
    
    delta = ppl_quant - ppl_orig
    rel_percent = (delta / ppl_orig) * 100
    
    if rel_percent <= thresholds.get('excellent', 1.0):
        quality = 'excellent'
    elif rel_percent <= thresholds.get('acceptable', 5.0):
        quality = 'acceptable'
    else:
        quality = 'poor'
        
    return {
        'perplexity_original': round(float(ppl_orig), 4),
        'perplexity_quantized': round(float(ppl_quant), 4),
        'perplexity_delta': round(float(delta), 4),
        'relative_delta_percent': round(float(rel_percent), 4),
        'quality': quality
    }
