import numpy as np

def ema_weights(ema_w: np.ndarray, current_w: np.ndarray, decay: float = 0.999) -> np.ndarray:
    ema_w = np.array(ema_w, dtype=float)
    current_w = np.array(current_w, dtype=float)
    
    new_ema = decay * ema_w + (1.0 - decay) * current_w
    return np.round(new_ema, 4)
