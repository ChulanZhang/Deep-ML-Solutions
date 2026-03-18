import numpy as np

def infonce_loss(z_i: np.ndarray, z_j: np.ndarray, temperature: float = 0.1) -> float:
    z_i = np.array(z_i, dtype=float)
    z_j = np.array(z_j, dtype=float)
    
    z_i = z_i / (np.linalg.norm(z_i, axis=-1, keepdims=True) + 1e-8)
    z_j = z_j / (np.linalg.norm(z_j, axis=-1, keepdims=True) + 1e-8)
    
    logits = np.dot(z_i, z_j.T) / temperature
    
    batch_size = z_i.shape[0]
    labels = np.arange(batch_size)
    
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    true_probs = probs[np.arange(batch_size), labels]
    loss = -np.mean(np.log(np.clip(true_probs, 1e-10, 1.0)))
    return float(np.round(loss, 4))
