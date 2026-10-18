import torch

def sobel_edge_detection(image: torch.Tensor) -> torch.Tensor:
    image = torch.tensor(image, dtype=float)
    h, w = image.shape
    out = torch.zeros_like(image)
    
    Gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = image[i-1:i+2, j-1:j+2]
            gx = torch.sum(patch * Gx)
            gy = torch.sum(patch * Gy)
            out[i, j] = torch.sqrt(gx**2 + gy**2)
            
    return torch.round(out, 4)
