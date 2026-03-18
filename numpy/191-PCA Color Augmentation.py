import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha_std: float = 0.1) -> np.ndarray:
    image = np.array(image, dtype=float)
    shape = image.shape
    img_flat = image.reshape(-1, 3) / 255.0
    
    mean = np.mean(img_flat, axis=0)
    img_centered = img_flat - mean
    
    cov = np.cov(img_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    alphas = np.random.normal(0, alpha_std, 3)
    delta = np.dot(eigenvectors, alphas * eigenvalues)
    
    img_aug = img_flat + delta
    img_aug = np.clip(img_aug * 255.0, 0, 255)
    
    return np.round(img_aug.reshape(shape), 4)
