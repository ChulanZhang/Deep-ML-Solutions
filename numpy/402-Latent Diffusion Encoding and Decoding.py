import numpy as np

def latent_diffusion_codec(x: np.ndarray, E: np.ndarray, D: np.ndarray) -> tuple:
    x = np.array(x, dtype=float)
    E = np.array(E, dtype=float)
    D = np.array(D, dtype=float)
    
    if x.ndim == 2:
        latent = np.dot(x, E)
        reconstructed = np.dot(latent, D)
    else:
        # standard 1D fallback
        latent = np.dot(x, E)
        reconstructed = np.dot(latent, D)
        
    return np.round(latent, 4), np.round(reconstructed, 4)
