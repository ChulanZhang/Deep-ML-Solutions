import numpy as np

def unet_time_embedding(time_steps: list, embed_dim: int) -> np.ndarray:
    t = np.array(time_steps, dtype=float)[:, None]
    half_dim = embed_dim // 2
    
    emb = np.log(10000.0) / (half_dim - 1) if half_dim > 1 else 0.0
    emb = np.exp(np.arange(half_dim) * -emb)
    
    emb = t * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    
    if embed_dim % 2 == 1:
        emb = np.pad(emb, ((0,0), (0,1)), mode='constant')
        
    return np.round(emb, 4)
