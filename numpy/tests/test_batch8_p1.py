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

mod84 = load_module("diag", "84-Convert Vector to Diagonal Matrix.py")
mod85 = load_module("col", "85-Find the column space of a matrix.py")
mod86 = load_module("resh", "86-Reshape Matrix.py")
mod87 = load_module("csr", "87-Implement Compressed Row Sparse Matrix (CSR) Format Conversion.py")
mod88 = load_module("ortho", "88-Implement Orthogonal Projection of a Vector onto a Line.py")

make_diagonal = mod84.make_diagonal
matrix_image = mod85.matrix_image
reshape_matrix = mod86.reshape_matrix
compressed_row_sparse_matrix = mod87.compressed_row_sparse_matrix
orthogonal_projection = mod88.orthogonal_projection

def check():
    # 84
    d = make_diagonal(np.array([1, 2, 3]))
    assert d == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    print("Diagonal OK")
    
    # 85
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    cs = matrix_image(A)
    # Output could be columns 0 and 1, testing dimensions
    assert len(cs) == 3 and len(cs[0]) == 2
    print("Col Space OK")
    
    # 86
    r = reshape_matrix([[1,2,3,4],[5,6,7,8]], (4, 2))
    assert r == [[1, 2], [3, 4], [5, 6], [7, 8]]
    print("Reshape OK")
    
    # 87
    dm = [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [3, 0, 4, 0],
        [1, 0, 0, 5]
    ]
    csr = compressed_row_sparse_matrix(dm)
    assert csr == ([1, 2, 3, 4, 1, 5], [0, 1, 0, 2, 0, 3], [0, 1, 2, 4, 6])
    print("CSR OK")
    
    # 88
    v = [3, 4]
    L = [1, 0]
    op = orthogonal_projection(v, L)
    assert op == [3.0, 0.0]
    print("Ortho Proj OK")

if __name__ == '__main__':
    check()
