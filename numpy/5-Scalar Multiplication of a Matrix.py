def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    """
    Multiply each element of a 2D matrix by a scalar.
    Returns a 2D tensor of the same shape.
    """
    return [[element * scalar for element in row] for row in matrix]