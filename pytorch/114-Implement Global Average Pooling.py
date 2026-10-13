import torch

def global_average_pooling(x):
    """
    Computes global average pooling for a 3D numpy array (H, W, C).
    Output shape: (channels,)
    """
    x = torch.tensor(x, dtype=float)
    return torch.mean(x, dim=(0, 1))
