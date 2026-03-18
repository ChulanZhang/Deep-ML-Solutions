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

mod98 = load_module("pool", "98-Overlapping Max Pooling.py")
mod99 = load_module("lrn", "99-Implement Local Response Normalization.py")
mod100 = load_module("pca", "100-PCA Color Augmentation.py")
mod101 = load_module("drop", "101-Dropout Layer.py")
mod102 = load_module("res", "102-Implement a Simple Residual Block with Shortcut Connection.py")

pool = mod98.overlapping_max_pool2d
lrn = mod99.local_response_normalization
pca = mod100.pca_color_augmentation
Dropout = mod101.DropoutLayer
residual_block = mod102.residual_block

def check():
    # 98
    x_pool = np.arange(1, 17).reshape(1, 1, 4, 4).astype(float)
    out_pool = pool(x_pool, 3, 2)
    assert np.allclose(out_pool[0, 0], [[11, 12], [15, 16]])
    print("Pool OK")
    
    # 99 LRN
    x_lrn = np.ones((1, 3, 2, 2))
    out_lrn = lrn(x_lrn, 5, 2.0, 1e-4, 0.75)
    assert out_lrn.shape == (1, 3, 2, 2)
    print("LRN OK")
    
    # 100 PCA
    np.random.seed(42)
    img = np.random.randint(0, 256, (4, 4, 3)).astype(np.uint8)
    alpha = np.array([0.1, 0.2, 0.3])
    out_pca = pca(img, alpha)
    assert out_pca.shape == (4, 4, 3)
    print("PCA OK")
    
    # 101 Dropout
    d = Dropout(0.5)
    x_d = np.ones((10, 10))
    out_d = d.forward(x_d, True)
    assert out_d.shape == (10, 10)
    out_back = d.backward(np.ones_like(x_d))
    assert out_back.shape == (10,10)
    print("Dropout OK")
    
    # 102 Residual
    x_r = np.array([1.0, 2.0])
    w1 = np.eye(2)
    w2 = 0.5 * np.eye(2)
    out_r = residual_block(x_r, w1, w2)
    assert np.allclose(out_r, [1.5, 3.0])
    print("Residual OK")

if __name__ == '__main__':
    check()
