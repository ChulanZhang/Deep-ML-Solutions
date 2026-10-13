import torch

def pca_color_augmentation(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Apply PCA color augmentation to an RGB image.
    """
    image_float = image.to(torch.float32)
    H, W, C = image_float.shape
    
    # Reshape to (N, 3)
    pixels = image_float.reshape(-1, 3)
    
    # PCA
    mean = torch.mean(pixels, dim=0)
    centered = pixels - mean
    cov = np.cov(centered, rowvar=False)
    
    # Eigen decomposition
    evals, evecs = torch.linalg.eigh(cov)
    
    # Delta
    delta = torch.zeros(3)
    for i in range(3):
        # AlexNet paper formula: add [p1, p2, p3][alpha1*lambda1, alpha2*lambda2, alpha3*lambda3]^T
        delta += alpha[i] * evals[i] * evecs[:, i]
        
    aug_image = image_float + delta
    aug_image = torch.clamp(aug_image, 0, 255)
    
    return aug_image
