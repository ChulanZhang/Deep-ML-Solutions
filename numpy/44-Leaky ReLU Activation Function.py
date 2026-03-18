def leaky_relu(z: float, alpha: float = 0.01) -> float:
    z = float(z)
    return z if z > 0 else z * alpha
