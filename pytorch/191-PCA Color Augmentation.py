import torch

def pca_color_augmentation(image: torch.Tensor, alpha_std: float = 0.1) -> torch.Tensor:
    image = torch.tensor(image, dtype=float)
    shape = image.shape
    img_flat = image.reshape(-1, 3) / 255.0
    
    mean = torch.mean(img_flat, dim=0)
    img_centered = img_flat - mean
    
    cov = np.cov(img_centered.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    alphas = np.random.normal(0, alpha_std, 3)
    delta = torch.matmul(eigenvectors, alphas * eigenvalues)
    
    img_aug = img_flat + delta
    img_aug = torch.clamp(img_aug * 255.0, 0, 255)
    
    return torch.round(img_aug.reshape(shape), 4)
