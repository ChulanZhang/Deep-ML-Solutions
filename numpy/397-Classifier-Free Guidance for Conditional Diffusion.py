import numpy as np

def classifier_free_guidance(uncond_noise: np.ndarray, cond_noise: np.ndarray, guidance_scale: float) -> np.ndarray:
    uncond = np.array(uncond_noise, dtype=float)
    cond = np.array(cond_noise, dtype=float)
    
    noise = uncond + guidance_scale * (cond - uncond)
    return np.round(noise, 4)
