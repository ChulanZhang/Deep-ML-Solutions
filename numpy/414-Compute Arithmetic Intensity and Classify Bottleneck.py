import numpy as np

def arithmetic_intensity(flops: float, bytes_accessed: float) -> tuple:
    ai = flops / (bytes_accessed + 1e-12)
    # Hardware threshold example ~ 100 FLOPS/Byte
    if ai < 100:
        bottleneck = "Memory-Bound"
    else:
        bottleneck = "Compute-Bound"
    return np.round(ai, 4), bottleneck
