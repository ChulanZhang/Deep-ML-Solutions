import torch

def clip_gradients_value(grads: list, clip_val: float) -> list:
    clipped = []
    for g in grads:
        g_arr = torch.tensor(g, dtype=float)
        clipped.append(torch.clamp(g_arr, -clip_val, clip_val).tolist())
    return clipped
