import numpy as np

def hardtanh(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    return np.clip(x, -1.0, 1.0)
