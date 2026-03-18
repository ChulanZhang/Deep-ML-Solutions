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

mod95 = load_module("gs", "95-Gauss-Seidel Method for Solving Linear Systems.py")
mod96 = load_module("svd", "96-Singular Value Decomposition (SVD) of 2x2 Matrix.py")
mod97 = load_module("det", "97-Determinant of a 4x4 Matrix using Laplace's Expansion.py")

gauss_seidel = mod95.gauss_seidel
svd_2x2_singular_values = mod96.svd_2x2_singular_values
determinant_4x4 = mod97.determinant_4x4

def check():
    # Gauss Seidel
    A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
    b = [-1, 2, 3]
    x = gauss_seidel(A, b, 2)
    assert len(x) == 3
    print("Gauss-Seidel OK")
    
    # SVD 2x2
    A_svd = np.array([[1, 2], [3, 4]])
    U, S, Vt = svd_2x2_singular_values(A_svd)
    # Check reconstruction
    A_recon = U @ np.diag(S) @ Vt
    assert np.allclose(A_svd, A_recon)
    print("SVD 2x2 OK")
    
    # Determinant
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    det = determinant_4x4(matrix)
    # Det of this specific linearly dependent matrix is 0
    assert np.isclose(det, 0.0)
    print("Determinant OK")

if __name__ == '__main__':
    check()
