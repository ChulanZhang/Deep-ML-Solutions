import torch

def unet_time_embedding(time_steps: list, embed_dim: int) -> torch.Tensor:
    t = torch.tensor(time_steps, dtype=float)[:, None]
    half_dim = embed_dim // 2
    
    emb = torch.log(10000.0) / (half_dim - 1) if half_dim > 1 else 0.0
    emb = torch.exp(torch.arange(half_dim) * -emb)
    
    emb = t * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    if embed_dim % 2 == 1:
        emb = np.pad(emb, ((0,0), (0,1)), mode='constant')
        
    return torch.round(emb, 4)
