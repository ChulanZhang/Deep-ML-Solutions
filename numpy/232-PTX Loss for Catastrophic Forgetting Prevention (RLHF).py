import numpy as np

def ptx_loss(logits: np.ndarray, labels: np.ndarray, ptx_coef: float = 1.0) -> float:
    logits = np.array(logits, dtype=float)
    labels = np.array(labels, dtype=int)
    
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    if logits.ndim == 2:
        # (batch, vocab)
        batch_idx = np.arange(labels.shape[0])
        true_probs = probs[batch_idx, labels]
    else:
        batch_idx, seq_idx = np.indices(labels.shape)
        true_probs = probs[batch_idx, seq_idx, labels]
        
    ce_loss = -np.log(np.clip(true_probs, 1e-10, 1.0))
    return float(ptx_coef * np.mean(ce_loss))
