import numpy as np

def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
    a = np.array(anchor, dtype=float)
    p = np.array(positive, dtype=float)
    n = np.array(negative, dtype=float)
    
    d_ap = np.linalg.norm(a - p, axis=-1)
    d_an = np.linalg.norm(a - n, axis=-1)
    
    loss = np.maximum(0.0, d_ap - d_an + margin)
    return float(np.round(np.mean(loss), 4))
