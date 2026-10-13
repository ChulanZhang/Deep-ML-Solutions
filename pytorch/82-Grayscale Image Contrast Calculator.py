import torch

def calculate_contrast(img):
    img_arr = torch.tensor(img)
    if img_arr.size == 0:
        return 0.0
    return float(torch.amax(img_arr) - torch.amin(img_arr))
