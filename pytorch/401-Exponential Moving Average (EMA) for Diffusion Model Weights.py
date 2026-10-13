import torch

def ema_weights(ema_w: torch.Tensor, current_w: torch.Tensor, decay: float = 0.999) -> torch.Tensor:
    ema_w = torch.tensor(ema_w, dtype=float)
    current_w = torch.tensor(current_w, dtype=float)
    
    new_ema = decay * ema_w + (1.0 - decay) * current_w
    return torch.round(new_ema, 4)
