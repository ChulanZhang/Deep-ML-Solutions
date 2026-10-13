import numpy as np

def translate_object(points, tx, ty):
    """
    Apply a 2D translation matrix to a set of points.
    """
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)
    
    translated_points = []
    for point in points:
        # Convert to homogeneous coordinates
        p_homo = np.array([point[0], point[1], 1.0])
        # Multiply by translation matrix
        p_trans = translation_matrix @ p_homo
        # Convert back
        translated_points.append([p_trans[0], p_trans[1]])
        
    return translated_points
