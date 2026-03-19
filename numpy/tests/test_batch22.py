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

mod453 = load_module("m453", "453-Implement Graph Convolution Network (GCN) Layer.py")
mod454 = load_module("m454", "454-Autoregressive Video Chunk FPS Calculator.py")
mod455 = load_module("m455", "455-Historical Context Compression Ratio.py")
mod456 = load_module("m456", "456-Multi-term Memory Patchification for Video.py")
mod457 = load_module("m457", "457-Implement Attention Sink Detection.py")

def check():
    A = np.array([[0, 1], [1, 0]])
    H = np.random.randn(2, 4)
    W = np.random.randn(4, 2)
    gcn = mod453.gcn_layer(A, H, W)
    assert gcn.shape == (2, 2)
    
    fps = mod454.video_chunk_fps(100, 10, 0.5)
    assert fps == 20.0
    
    cr = mod455.context_compression_ratio(1024, 128)
    assert cr == 8.0
    
    vid = np.random.randn(8, 3, 16, 16)
    patch = mod456.video_memory_patchify(vid, 2, 8, 8)
    assert patch.shape == (4*2*2, 2*3*8*8)
    
    attn = np.random.rand(2, 4, 10, 10)
    # Norm softmax over last dim
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    snk = mod457.detect_attention_sinks(attn, 2, 0.1)
    assert isinstance(snk, bool)
    
    print("Batch 22 OK")

if __name__ == '__main__':
    check()
