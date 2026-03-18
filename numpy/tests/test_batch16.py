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

mod236 = load_module("m236", "236-Mean Ablation for Circuit Discovery.py")
mod237 = load_module("m237", "237-Convert RGB Image to Grayscale.py")
mod238 = load_module("m238", "238-Flip an Image Horizontally or Vertically.py")
mod239 = load_module("m239", "239-Apply Zero Padding to an Image.py")
mod240 = load_module("m240", "240-Bilinear Image Resizing.py")
mod241 = load_module("m241", "241-Sobel Edge Detection.py")
mod242 = load_module("m242", "242-Non-Maximum Suppression for Object Detection.py")
mod262 = load_module("m262", "262-Implement the Mish Activation Function.py")
mod263 = load_module("m263", "263-Implement Binary Cross-Entropy Loss.py")
mod264 = load_module("m264", "264-Implement the Tanh Activation Function.py")
mod265 = load_module("m265", "265-Implement 2D Average Pooling.py")
mod266 = load_module("m266", "266-Implement the Hardtanh Activation Function.py")
mod267 = load_module("m267", "267-Implement Neural Memory Update with Surprise and Momentum.py")
mod271 = load_module("m271", "271-Implement Gated Attention.py")
mod287 = load_module("m287", "287-Implement GRU Cell.py")

def check():
    # 236
    r236 = mod236.mean_ablation([1.0, 2.0, 3.0], [0])
    assert len(r236) == 3
    # 237
    r237 = mod237.rgb_to_grayscale(np.random.rand(2, 2, 3))
    assert r237.shape == (2, 2)
    # 238
    r238 = mod238.flip_image(np.random.rand(2, 4), "horizontal")
    assert r238.shape == (2, 4)
    # 239
    r239 = mod239.zero_padding(np.ones((2, 2)), 1)
    assert r239.shape == (4, 4)
    # 240
    r240 = mod240.bilinear_resize(np.zeros((4, 4, 3)), 2, 2)
    assert r240.shape == (2, 2, 3)
    # 241
    r241 = mod241.sobel_edge_detection(np.random.rand(4, 4))
    assert r241.shape == (4, 4)
    # 242
    r242 = mod242.nms([[0,0,10,10], [1,1,11,11], [10,10,20,20]], [0.9, 0.8, 0.7], 0.5)
    assert len(r242) == 2
    # 262
    r262 = mod262.mish([1.0, 0.0, -1.0])
    assert len(r262) == 3
    # 263
    r263 = mod263.binary_cross_entropy([1, 0], [0.9, 0.1])
    assert isinstance(r263, float)
    # 264
    r264 = mod264.tanh([1.0, 0.0])
    assert len(r264) == 2
    # 265
    r265 = mod265.average_pooling(np.random.rand(4, 4), 2, 2)
    assert r265.shape == (2, 2)
    # 266
    r266 = mod266.hardtanh([2.0, -2.0, 0.5])
    assert np.all(r266 == [1.0, -1.0, 0.5])
    # 267
    r267 = mod267.neural_memory_update([0.5], [1.0], 0.2, 0.8)
    assert len(r267) == 1
    # 271
    r271 = mod271.gated_attention(np.random.rand(2, 4), np.random.rand(4, 3), np.random.rand(4, 3))
    assert r271.shape == (2, 3)
    # 287
    r287 = mod287.gru_cell(np.random.rand(2, 4), np.random.rand(2, 3), 
                           np.random.rand(4, 3), np.random.rand(3, 3), np.random.rand(3),
                           np.random.rand(4, 3), np.random.rand(3, 3), np.random.rand(3),
                           np.random.rand(4, 3), np.random.rand(3, 3), np.random.rand(3))
    assert r287.shape == (2, 3)
    
    print("Batch 16 OK")

if __name__ == '__main__':
    check()
