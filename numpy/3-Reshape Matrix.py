import numpy as np

from typing import Union, List, Tuple

def reshape_matrix(a: List[List[Union[int, float]]], new_shape: Tuple[int, int]) -> List[List[Union[int, float]]]:
    """
    Reshapes a given matrix into a specified shape.
    Returns [] if it cannot be reshaped.
    """
    # Check total elements
    if not a or not a[0]:
        total_elements = 0
    else:
        total_elements = len(a) * len(a[0])
        
    if total_elements != new_shape[0] * new_shape[1]:
        return []
        
    arr = np.array(a)
    reshaped = arr.reshape(new_shape)
    return reshaped.tolist()
