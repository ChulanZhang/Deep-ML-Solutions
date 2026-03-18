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

# 158-167 Bandits and Returns
mod158 = load_module("m158", "158-Epsilon-Greedy Action Selection for n-Armed Bandit.py")
mod159 = load_module("m159", "159-Incremental Mean for Online Reward Estimation.py")
mod161 = load_module("m161", "161-Exponential Weighted Average of Rewards.py")
mod162 = load_module("m162", "162-Upper Confidence Bound (UCB) Action Selection.py")
mod163 = load_module("m163", "163-Gradient Bandit Action Selection.py")
mod164 = load_module("m164", "164-Gambler\'s Problem: Value Iteration.py")
mod165 = load_module("m165", "165-Compute Discounted Return.py")
mod166 = load_module("m166", "166-Evaluate Expected Value in a Markov Decision Process.py")
mod167 = load_module("m167", "167-Calculate the Discounted Return for a Given Trajectory.py")

# 172-189
mod172 = load_module("m172", "172-Muon Optimizer Update with Newton-Schulz Iteration.py")
mod174 = load_module("m174", "174-Train a Simple GAN on 1D Gaussian Data.py")
mod177 = load_module("m177", "177-Implement MuonClip (qk-clip) for Stabilizing Attention.py")
mod178 = load_module("m178", "178-Implement Position-wise Feed-Forward Block with Residual and Dropout.py")
mod185 = load_module("m185", "185-Optical Flow EPE with Masks (OmniWorld-style metric).py")
mod189 = load_module("m189", "189-Implement Local Response Normalization (LRN).py")

def check():
    assert mod158.epsilon_greedy([0.5, 2.3, 1.7], 0.0) == 1
    assert mod159.incremental_mean(2.0, 2, 6.0) == 4.0
    assert mod161.exp_weighted_average(2.0, [5.0, 9.0], 0.3) == 4.73
    assert mod162.ucb_action([1, 1, 1, 1], [1.0, 2.0, 1.5, 0.5], 4, 2.0) == 1
    
    gb = mod163.GradientBandit(3, 0.1)
    a = gb.select_action()
    gb.update(a, 1.0)
    
    V, pi = mod164.gambler_value_iteration(0.4)
    assert len(V) == 101
    assert mod165.discounted_return([1, 1, 1], 0.5) == 1.75
    
    assert mod166.expected_action_value(0, 'a', {0: {'a': {0: 0.5, 1: 0.5}}}, {}, [1.0, 2.0], 0.9) == 1.35
    assert mod167.discounted_return([1, 2, 3, 4], 0.9) == 8.146
    
    th, B, O = mod172.muon_update(np.ones((3,3)), np.arange(1, 10).reshape(3,3), np.zeros((3,3)), 0.9, 0.01)
    
    gen = mod174.train_gan(4.0, 1.25)
    
    _, _, _ = mod177.muonclip_qk_clip([[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]], [[[1,0],[0,1]]], 1.0)
    
    ff = mod178.ffn([1.0, -1.0], [[1,2],[3,4]], [0.5,-0.5], [[2,1],[0.5,1]], [0,0.5], 0.0)
    assert np.allclose(ff, [1.0, -0.5])
    
    epe = mod185.flow_epe([[[1,0],[0,1]], [[-1,0],[0,-1]]], [[[0,0],[0,0]], [[0,0],[0,0]]])
    assert epe == 1.0
    
    x = np.random.randn(1, 3, 2, 2)
    lrn = mod189.local_response_normalization(x)
    assert lrn.shape == (1, 3, 2, 2)
    
    print("Batch 14 OK")

if __name__ == '__main__':
    check()
