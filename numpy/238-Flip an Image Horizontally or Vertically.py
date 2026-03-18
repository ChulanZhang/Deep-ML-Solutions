import numpy as np

def flip_image(img: np.ndarray, direction: str) -> np.ndarray:
    img = np.array(img)
    if direction == "horizontal":
        return img[:, ::-1]
    elif direction == "vertical":
        return img[::-1, :]
    else:
        return img
