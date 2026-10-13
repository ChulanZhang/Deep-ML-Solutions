import numpy as np

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    v = np.array(v, dtype=float)
    L = np.array(L, dtype=float)
    
    L_dot_L = np.dot(L, L)
    if L_dot_L == 0:
        return [0.0] * len(v)
        
    scalar = np.dot(v, L) / L_dot_L
    proj = scalar * L
    
    return [round(x, 3) for x in proj]
