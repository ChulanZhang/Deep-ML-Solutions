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

mod116 = load_module("lpe", "116-Learned Positional Embeddings.py")
mod117 = load_module("lm_tok", "117-Language Modeling Tokenizer Lab.py")
mod118 = load_module("rms", "118-RMSNorm.py")
mod119 = load_module("rope", "119-RoPE.py")
mod120 = load_module("kvc", "120-KV Cache.py")

learned_positional_encoding = mod116.learned_positional_encoding
train_tokenizer = mod117.train_tokenizer
rmsnorm = mod118.rmsnorm
apply_rope = mod119.apply_rope
kv_cache_attention_step = mod120.kv_cache_attention_step

def check():
    # 116 LPE
    tokens = np.ones((2, 3, 4))
    table = np.ones((10, 4))
    out_lpe = learned_positional_encoding(tokens, table, 1)
    assert out_lpe.shape == (2, 3, 4)
    print("LPE OK")
    
    # 117 Tokenizer Lab
    corpus = ["abc", "bcd"]
    enc, dec = train_tokenizer(corpus, 4)
    idx = enc("abcd")
    assert dec(idx) == "abcd"
    print("Tokenizer Lab OK")
    
    # 118 RMSNorm
    x_rms = np.ones((2, 4))
    g = np.ones(4)
    out_rms = rmsnorm(x_rms, g)
    assert out_rms.shape == (2, 4)
    print("RMSNorm OK")
    
    # 119 RoPE
    x_rope = np.ones((3, 4))
    pos = np.array([0, 1, 2])
    out_rope = apply_rope(x_rope, pos)
    assert out_rope.shape == (3, 4)
    print("RoPE OK")
    
    # 120 KV Cache
    x_new = np.ones(4)
    W_Q, W_K, W_V = np.ones((4, 2)), np.ones((4, 2)), np.ones((4, 3))
    out_kv, cache = kv_cache_attention_step(x_new, W_Q, W_K, W_V)
    assert out_kv.shape == (3,)
    out_kv_2, cache_2 = kv_cache_attention_step(x_new, W_Q, W_K, W_V, cache)
    assert out_kv_2.shape == (3,)
    assert cache_2[0].shape == (2, 2)
    print("KV Cache OK")

if __name__ == '__main__':
    check()
