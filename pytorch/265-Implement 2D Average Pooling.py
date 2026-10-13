import torch

def average_pooling(image: torch.Tensor, pool_size: int, stride: int) -> torch.Tensor:
    image = torch.tensor(image, dtype=float)
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    out = torch.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            val = torch.mean(image[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size])
            out[i, j] = val
    return torch.round(out, 4)
