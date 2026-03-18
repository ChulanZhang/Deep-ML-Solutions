import numpy as np

def temperature_decay(initial_temp: float, decay_rate: float, step: int) -> float:
    return float(np.round(initial_temp * (decay_rate ** step), 4))
