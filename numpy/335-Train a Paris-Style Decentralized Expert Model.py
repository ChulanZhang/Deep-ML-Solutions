import numpy as np

def paris_expert_consensus(expert_predictions: list) -> np.ndarray:
    preds = np.array(expert_predictions, dtype=float)
    consensus = np.mean(preds, axis=0)
    return np.round(consensus, 4)
