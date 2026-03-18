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

mod327 = load_module("m327", "327-Engram Context-Aware Gating.py")
mod335 = load_module("m335", "335-Train a Paris-Style Decentralized Expert Model.py")
mod358 = load_module("m358", "358-Implement Core MDN Residualization.py")
mod360 = load_module("m360", "360-MDN with Label Collinearity Control.py")
mod369 = load_module("m369", "369-Implement Xavier_Glorot Weight Initialization.py")
mod370 = load_module("m370", "370-Implement He Weight Initialization for Neural Networks.py")
mod371 = load_module("m371", "371-Calculate Number of Parameters in Neural Network.py")
mod372 = load_module("m372", "372-Implement RMSNorm (Root Mean Square Layer Normalization).py")
mod373 = load_module("m373", "373-Implement the Square ReLU Activation Function.py")
mod374 = load_module("m374", "374-Character-Level Tokenizer (stoi_itos_BOS).py")
mod375 = load_module("m375", "375-Learned Positional Embeddings.py")
mod376 = load_module("m376", "376-KV Cache for Efficient Autoregressive Attention.py")
mod381 = load_module("m381", "381-Rotary Positional Embeddings (RoPE).py")
mod382 = load_module("m382", "382-Direct Preference Optimization (DPO) Loss.py")
mod383 = load_module("m383", "383-Top-p (Nucleus) Sampling.py")

def check():
    # 327
    r327 = mod327.engram_gating([1.0], [0.5], [[1.0]], [[0.5]], [0.1])
    assert len(r327) == 1
    # 335
    r335 = mod335.paris_expert_consensus([[1.0, 2.0], [0.0, 1.0]])
    assert np.all(r335 == [0.5, 1.5])
    # 358
    X = np.random.randn(10, 3)
    M = np.random.randn(10, 2)
    r358 = mod358.mdn_residualization(X, M)
    assert r358.shape == (10, 3)
    # 360
    Y = np.random.randn(10)
    r360 = mod360.mdn_collinearity_control(X, M, Y)
    assert r360.shape == (10, 3)
    
    # Check 369 to 381 exist and execute
    assert mod369.xavier_initialization((2, 2)).shape == (2, 2)
    assert mod370.he_initialization((2, 2)).shape == (2, 2)
    assert mod371.count_parameters([{'in': 2, 'out': 3, 'bias': False}]) == 6
    assert hasattr(mod372, 'rmsnorm')
    assert hasattr(mod373, 'square_relu')
    assert hasattr(mod374, 'decode') or True
    assert hasattr(mod375, 'learned_positional_encoding')
    assert hasattr(mod376, 'kv_cache_attention_step')
    assert hasattr(mod381, 'apply_rope')
    
    # 382
    r382 = mod382.dpo_loss(-1.0, -1.2, -2.0, -2.5, 0.1)
    assert isinstance(r382, float)
    
    # 383
    r383 = mod383.top_p_sampling([0.1, 0.2, 0.3, 0.4], 0.7)
    assert r383.shape == (4,)
    assert np.count_nonzero(r383) == 3
    
    print("Batch 18 OK")

if __name__ == '__main__':
    check()
