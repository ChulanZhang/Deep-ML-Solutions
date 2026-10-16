import torch

def flip_image(img: torch.Tensor, direction: str) -> torch.Tensor:
    img = torch.tensor(img)
    if direction == "horizontal":
        return img[:, ::-1]
    elif direction == "vertical":
        return img[::-1, :]
    else:
        return img
