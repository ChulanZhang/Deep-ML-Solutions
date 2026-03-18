import sys
import numpy as np
import importlib.util
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
numpy_dir = os.path.dirname(script_dir)

def load_module(name, filename):
    path = os.path.join(numpy_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod92 = load_module("rref", "92-Implement Reduced Row Echelon Form (RREF) Function.py")
mod93 = load_module("jacobi", "93-Solve Linear Equations using Jacobi Method.py")
mod94 = load_module("trans", "94-2D Translation Matrix Implementation.py")

rref = mod92.rref
solve_jacobi = mod93.solve_jacobi
translate_object = mod94.translate_object

def check():
    # RREF
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    r = rref(matrix)
    # RREF of above is [[1, 0, -1], [0, 1, 2], [0, 0, 0]]
    assert np.allclose(r[0], [1, 0, -1])
    assert np.allclose(r[1], [0, 1, 2])
    print("RREF OK")
    
    # Jacobi
    A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
    b = [-1, 2, 3]
    x = solve_jacobi(A, b, 2)
    assert len(x) == 3
    print("Jacobi OK")
    
    # 2D Translation
    points = [[0, 0], [1, 0], [0.5, 1]]
    tx, ty = 2, 3
    t_points = translate_object(points, tx, ty)
    assert t_points == [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
    print("2D Translate OK")

if __name__ == '__main__':
    check()
