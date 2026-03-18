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

mod56 = load_module("kldiv", "56-KL Divergence Between Two Normal Distributions.py")
mod19 = load_module("pca", "19-Principal Component Analysis (PCA) Implementation.py")
mod17 = load_module("kmeans", "17-K-Means Clustering.py")

kl_divergence_normal = mod56.kl_divergence_normal
pca = mod19.pca
k_means_clustering = mod17.k_means_clustering

def check():
    # KL Divergence
    kl = kl_divergence_normal(0.0, 1.0, 1.0, 1.0)
    assert abs(kl - 0.5) < 1e-4
    print("KL Divergence OK")
    
    # PCA
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pcs = pca(data, 1)
    assert pcs.shape == (2, 1)
    print("PCA OK")
    
    # K-Means
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10
    final = k_means_clustering(points, k, initial_centroids, max_iterations)
    assert len(final) == 2
    print("K-Means OK")

if __name__ == '__main__':
    check()
