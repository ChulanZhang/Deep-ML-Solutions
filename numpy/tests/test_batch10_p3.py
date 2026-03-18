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

mod106 = load_module("group", "106-Implement Group Normalization.py")
mod107 = load_module("mdn_res", "107-Implement Core MDN Residualization.py")
mod108 = load_module("mdn_coll", "108-Label Collinearity Control.py")
mod110 = load_module("dcor", "110-Distance Correlation Squared.py")

group_normalization = mod106.group_normalization
mdn_res = mod107.mdn_residualization
mdn_coll = mod108.mdn_with_collinearity
distance_correlation_squared = mod110.distance_correlation_squared

try:
    mod109 = load_module("deconf", "109-Feature Deconfounder Lab.py")
    FeatureDeconfounder = mod109.FeatureDeconfounder
except ImportError:
    FeatureDeconfounder = None

def check():
    # 106 Group Norm
    X = np.ones((2, 4, 2, 2))
    gamma = np.ones(4)
    beta = np.zeros(4)
    out = group_normalization(X, gamma, beta, 2)
    assert out.shape == (2, 4, 2, 2)
    print("Group Norm OK")
    
    # 107 MDN Residualization
    f = np.array([[1.0], [2.0], [3.0]])
    X_mdn = np.array([[1.0], [1.0], [1.0]])
    inv = np.linalg.inv(X_mdn.T @ X_mdn)
    res = mdn_res(f, X_mdn, inv, 3)
    assert res.shape == (3, 1)
    print("MDN Residualization OK")
    
    # 108 Collinearity
    y = np.array([[1], [0], [1]])
    inv_tilde = np.linalg.inv(np.concatenate([X_mdn, y], axis=1).T @ np.concatenate([X_mdn, y], axis=1) + np.eye(2)*0.01)
    res_coll = mdn_coll(f, X_mdn, y, inv_tilde, 3)
    assert res_coll.shape == (3, 1) # Should execute correctly mathematically
    print("Label Collinearity OK")
    
    # 109 Feature Deconfounder
    if FeatureDeconfounder is not None:
        import torch
        fd = FeatureDeconfounder()
        fd.fit(torch.tensor(X_mdn, dtype=torch.float32))
        res_fd = fd.transform(torch.tensor(f, dtype=torch.float32), torch.tensor(X_mdn, dtype=torch.float32))
        assert res_fd.shape == (3, 1)
        print("Feature Deconfounder OK")
    else:
        print("Feature Deconfounder SKIPPED (torch not installed)")
    
    # 110 Distance Correlation
    A = np.array([[1], [2], [3]])
    B = np.array([[1], [2], [3]])
    dcor = distance_correlation_squared(A, B)
    assert 0 <= dcor <= 1.01
    print("Distance Correlation OK")

if __name__ == '__main__':
    check()
