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

mod190 = load_module("m190", "190-Overlapping Max Pooling.py")
mod191 = load_module("m191", "191-PCA Color Augmentation.py")
mod208 = load_module("m208", "208-Flash Attention v1 - Forward Pass.py")
mod216 = load_module("m216", "216-Thanksgiving Feast Predictor: Softmax for Dish Selection.py")
mod217 = load_module("m217", "217-Derivatives of Activation Functions.py")
mod218 = load_module("m218", "218-Compute the Hessian Matrix.py")
mod222 = load_module("m222", "222-LoRA: Low-Rank Adaptation Forward Pass.py")
mod223 = load_module("m223", "223-QLoRA: Quantized Low-Rank Adaptation Forward Pass.py")
mod229 = load_module("m229", "229-Sparse MoE Top-K Routing.py")
mod230 = load_module("m230", "230-3D CNN Forward Pass Implementation.py")
mod231 = load_module("m231", "231-Temperature Decay Scheduler.py")
mod232 = load_module("m232", "232-PTX Loss for Catastrophic Forgetting Prevention (RLHF).py")
mod233 = load_module("m233", "233-Inference Head Pruning for Transformers.py")
mod234 = load_module("m234", "234-Block-wise FP8 Quantization.py")
mod235 = load_module("m235", "235-Implement the SGTM Parameter Update Step.py")

def check():
    # 190
    img = np.random.rand(4, 4)
    assert mod190.max_pooling(img, 2, 2).shape == (2, 2)
    # 191
    img_color = np.random.rand(8, 8, 3) * 255
    assert mod191.pca_color_augmentation(img_color).shape == (8, 8, 3)
    # 208
    q = np.random.randn(2, 4); k = np.random.randn(2, 4); v = np.random.randn(2, 4)
    assert mod208.flash_attention_forward(q, k, v).shape == (2, 4)
    # 216
    assert len(mod216.feast_predictor([1, 2, 3])) == 3
    # 217
    assert len(mod217.activation_derivatives([1.0, -1.0], 'relu')) == 2
    # 218
    h_mat = mod218.compute_hessian(lambda x: x[0]**2 + x[1]**2, [1.0, 2.0])
    assert h_mat.shape == (2, 2)
    # 222
    lora = mod222.lora_forward(np.eye(2), np.eye(2), np.eye(2), np.ones((2, 2)))
    assert np.array(lora).shape == (2, 2)
    # 223
    assert hasattr(mod223, 'qlora_forward')
    # 229
    routing, top_idx = mod229.topk_routing(np.random.randn(2, 4), np.random.randn(4, 3), 2)
    assert routing.shape == (2, 2)
    # 230
    conv3d_res = mod230.conv3d(np.random.randn(4, 4, 4), np.random.randn(2, 2, 2))
    assert conv3d_res.shape == (3, 3, 3)
    # 231
    assert mod231.temperature_decay(1.0, 0.9, 2) == 0.81
    # 232
    ptx = mod232.ptx_loss(np.random.randn(2, 5), np.array([1, 2]))
    assert isinstance(ptx, float)
    # 233
    wq = np.random.randn(4, 4); wk = np.random.randn(4, 4); wv = np.random.randn(4, 4); wo = np.random.randn(4, 4)
    q_new, k_new, v_new, o_new = mod233.prune_heads(wq, wk, wv, wo, [0], 2)
    assert q_new.shape == (4, 2)
    assert o_new.shape == (2, 4)
    # 234
    assert hasattr(mod234, 'fp8_block_quantize')
    # 235
    p, m = mod235.sgtm_update(1.0, 0.1, 0.1)
    assert p < 1.0
    
    print("Batch 15 OK")

if __name__ == '__main__':
    check()
