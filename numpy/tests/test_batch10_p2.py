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

mod103 = load_module("gap", "103-Implement Global Average Pooling.py")
mod104 = load_module("bn", "104-Batch Normalization for BCHW Input.py")
mod105 = load_module("dense", "105-Implement a Dense Block with 2D Convolutions.py")

global_average_pooling = mod103.global_average_pooling
batch_normalization = mod104.batch_normalization
dense_net_block = mod105.dense_net_block

def check():
    # 103 GAP
    x_gap = np.ones((10, 10, 3))
    out_gap = global_average_pooling(x_gap)
    assert np.allclose(out_gap, [1.0, 1.0, 1.0])
    print("GAP OK")
    
    # 104 Batch Norm
    X = np.ones((2, 3, 4, 4))
    gamma = np.ones(3)
    beta = np.zeros(3)
    out_bn, rm, rv = batch_normalization(X, gamma, beta)
    assert out_bn.shape == (2, 3, 4, 4)
    assert rm.shape == (3,)
    assert rv.shape == (3,)
    print("Batch Norm OK")
    
    # 105 Dense Block
    input_data = np.ones((1, 4, 4, 2)) # N, H, W, C
    num_layers = 1
    growth_rate = 2
    kernels = [np.ones((3, 3, 2, 2))]
    out_dense = dense_net_block(input_data, num_layers, growth_rate, kernels)
    assert out_dense.shape == (1, 4, 4, 4)
    print("Dense Block OK")

if __name__ == '__main__':
    check()
