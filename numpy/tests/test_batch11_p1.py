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

mod111 = load_module("mha", "111-Implement Multi-Head Attention.py")
mod112 = load_module("msa", "112-Implement Masked Self-Attention.py")
mod113 = load_module("ln", "113-Implement Layer Normalization for Sequence Data.py")
mod114 = load_module("pe", "114-Positional Encoding Calculator.py")
mod115 = load_module("char", "115-Character-Level Tokenizer.py")

multi_head_attention = mod111.multi_head_attention
compute_qkv = mod111.compute_qkv
masked_attention = mod112.masked_attention
layer_normalization = mod113.layer_normalization
pos_encoding = mod114.pos_encoding
CharTokenizer = mod115.CharTokenizer

def check():
    # 111 MHA
    np.random.seed(42)
    seq_len, d_model, n_heads = 4, 8, 2
    X = np.random.randn(seq_len, d_model)
    W_q, W_k, W_v = np.random.randn(d_model, d_model), np.random.randn(d_model, d_model), np.random.randn(d_model, d_model)
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    out_mha = multi_head_attention(Q, K, V, n_heads)
    assert out_mha.shape == (seq_len, d_model)
    print("MHA OK")
    
    # 112 Masked SA
    mask = np.tril(np.ones((seq_len, seq_len)))
    out_msa = masked_attention(Q, K, V, mask)
    assert out_msa.shape == (seq_len, d_model)
    print("Masked SA OK")
    
    # 113 LayerNorm
    X_ln = np.ones((2, 3, 4))
    gamma, beta = np.ones(4), np.zeros(4)
    out_ln = layer_normalization(X_ln, gamma, beta)
    assert out_ln.shape == (2, 3, 4)
    print("LayerNorm OK")
    
    # 114 Pos Encoding
    pe = pos_encoding(1, 4)
    assert pe.shape == (4,)
    print("Pos Encoding OK")
    
    # 115 Char Tokenizer
    text = "hello"
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode("helo")
    assert encoded[0] == 0 and encoded[-1] == 1
    decoded = tokenizer.decode(encoded)
    assert decoded == "<BOS>helo<EOS>"
    print("Char Tokenizer OK")

if __name__ == '__main__':
    check()
