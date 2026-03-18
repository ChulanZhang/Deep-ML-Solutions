import numpy as np

def clip_gradients_value(grads: list, clip_val: float) -> list:
    clipped = []
    for g in grads:
        g_arr = np.array(g, dtype=float)
        clipped.append(np.clip(g_arr, -clip_val, clip_val).tolist())
    return clipped
