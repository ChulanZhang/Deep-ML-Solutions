def scalar_multiply(matrix: torch.Tensor, scalar: int|float) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar.
    Returns a 2D tensor of the same shape.
    """
    return [[element * scalar for element in row] for row in matrix]