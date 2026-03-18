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

mod127 = load_module("prenorm", "127-Pre-Norm vs Post-Norm Transformer Block.py")
mod128 = load_module("gqa", "128-Grouped Query Attention.py")
mod129 = load_module("swa", "129-Sliding Window Attention.py")
mod130 = load_module("moe", "130-Mixture-of-Experts Gating.py")
mod131 = load_module("swi", "131-Swish SwiGLU Activation.py")

transformer_block = mod127.transformer_block
grouped_query_attention = mod128.grouped_query_attention
sliding_window_attention = mod129.sliding_window_attention
noisy_top_k_gating = mod130.noisy_top_k_gating
swiglu = mod131.swiglu

def check():
    np.random.seed(42)
    # 127 PreNorm
    x = np.ones((2, 4))
    W1 = np.ones((4, 4))
    b1 = np.zeros(4)
    W2 = np.ones((4, 4))
    b2 = np.zeros(4)
    g1 = np.ones(4)
    g2 = np.ones(4)
    out_pre = transformer_block(x, W1, b1, W2, b2, g1, b1, g2, b2, 'pre')
    out_post = transformer_block(x, W1, b1, W2, b2, g1, b1, g2, b2, 'post')
    assert out_pre.shape == (2, 4)
    assert out_post.shape == (2, 4)
    print("Pre-Norm vs Post-Norm OK")
    
    # 128 GQA
    Q = np.random.randn(2, 5, 8) # B=2, S=5, H_q=4, D=2 => 8
    K = np.random.randn(2, 5, 4) # B=2, S=5, H_kv=2, D=2 => 4
    V = np.random.randn(2, 5, 4)
    out_gqa = grouped_query_attention(Q, K, V, 4, 2)
    assert out_gqa.shape == (2, 5, 8)
    print("GQA OK")
    
    # 129 SWA
    Q_sw = np.ones((5, 3))
    K_sw = np.ones((5, 3))
    V_sw = np.ones((5, 4))
    out_swa = sliding_window_attention(Q_sw, K_sw, V_sw, 1)
    assert out_swa.shape == (5, 4)
    print("SWA OK")
    
    # 130 MoE
    X_moe = np.ones((2, 3))
    Wg = np.ones((3, 4))
    Wn = np.ones((3, 4))
    Nrms = np.random.randn(2, 4)
    out_moe = noisy_top_k_gating(X_moe, Wg, Wn, Nrms, 2)
    assert out_moe.shape == (2, 4)
    print("MoE Gating OK")
    
    # 131 SwiGLU
    x_swi = np.array([[-1.0, 2.0, 3.0, 4.0]]) # 2d = 4, d=2
    out_swi = swiglu(x_swi)
    assert out_swi.shape == (1, 2)
    print("SwiGLU OK")

if __name__ == '__main__':
    check()
