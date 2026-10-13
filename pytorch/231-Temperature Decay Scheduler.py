import torch

def temperature_decay(initial_temp: float, decay_rate: float, step: int) -> float:
    return float(torch.round(initial_temp * (decay_rate ** step), 4))
