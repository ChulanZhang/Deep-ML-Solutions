import numpy as np

def mean_ablation(activations: np.ndarray, indices: list) -> np.ndarray:
    activations = np.array(activations, dtype=float)
    mean_val = np.mean(activations)
    for idx in indices:
        activations[idx] = mean_val
    return np.round(activations, 4)
