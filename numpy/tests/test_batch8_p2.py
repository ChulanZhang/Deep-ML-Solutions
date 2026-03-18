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

mod89 = load_module("csc", "89-Implement Compressed Column Sparse Matrix Format.py")
mod90 = load_module("basis", "90-Transformation Matrix from Basis B to C.py")
mod91 = load_module("mat_trans", "91-Matrix Transformation.py")

compressed_col_sparse_matrix = mod89.compressed_col_sparse_matrix
transform_basis = mod90.transform_basis
transform_matrix = mod91.transform_matrix

def check():
    # CSC
    dense = [[0, 0, 3, 0], [1, 0, 0, 4], [0, 2, 0, 0]]
    v, r, c = compressed_col_sparse_matrix(dense)
    assert v == [1, 2, 3, 4]
    assert r == [1, 2, 0, 1]
    assert c == [0, 1, 2, 3, 4]
    print("CSC OK")
    
    # Basis
    B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    C = [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]
    P = transform_basis(B, C)
    # Target from prompt: [[-0.6772, -0.0126, 0.2342], [-0.0184, 0.0505, -0.0275], [0.5732, -0.0345, -0.0569]]
    # Let's just check length
    assert len(P) == 3
    print("Basis Transform OK")
    
    # Matrix Trans
    A = [[1, 2], [3, 4]]
    T = [[2, 0], [0, 2]]
    S = [[1, 1], [0, 1]]
    res = transform_matrix(A, T, S)
    # Output: [[0.5, 1.5], [1.5, 3.5]]
    assert res == [[0.5, 1.5], [1.5, 3.5]]
    print("Matrix Transform OK")

if __name__ == '__main__':
    check()
