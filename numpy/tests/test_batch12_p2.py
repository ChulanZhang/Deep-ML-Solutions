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

mod132 = load_module("i8", "132-Implement INT8 Quantization.py")
mod133 = load_module("fp8", "133-Block-wise FP8 Quantization.py")
mod134 = load_module("mx", "134-FP4 Quantization with Microscaling.py")
mod135 = load_module("sink", "135-Attention Sinks.py")
mod136 = load_module("grpo", "136-Implement GRPO Objective Function.py")
mod137 = load_module("adv", "137-Group Relative Advantage for GRPO.py")
mod138 = load_module("kl", "138-KL Divergence Estimator for GRPO.py")

int8_quantize = mod132.int8_quantize
fp8_block_quantize = mod133.fp8_block_quantize
mxfp4_quantize = mod134.mxfp4_quantize
detect_attention_sinks = mod135.detect_attention_sinks
grpo_objective = mod136.grpo_objective
compute_group_relative_advantage = mod137.compute_group_relative_advantage
kl_divergence_estimator = mod138.kl_divergence_estimator

def check():
    # 132 INT8
    res8 = int8_quantize([12.7, -6.35, 0.0])
    assert res8['scale'] > 0
    print("INT8 OK")
    
    # 133 FP8
    x_fp8 = np.ones((4, 4))
    q, s = fp8_block_quantize(x_fp8, block_size=16)
    assert q.shape == (4, 4)
    print("FP8 Block OK")
    
    # 134 MXFP4
    res_mxfp4 = mxfp4_quantize([1.5, -0.5, 3.2, 6.0])
    assert 'scales' in res_mxfp4
    print("MXFP4 OK")
    
    # 135 Sinks
    attn = np.ones((2, 5, 5)) # 2 heads
    sinks = detect_attention_sinks(attn, 0.5)
    assert len(sinks['sink_positions']) > 0
    print("Attn Sinks OK")
    
    # 136 GRPO Obj
    r = [1.0, 1.2]
    A = [0.5, -0.5]
    p_old = [0.1, 0.9]
    p_ref = [0.1, 0.9]
    obj = grpo_objective(r, A, p_old, p_ref)
    assert isinstance(obj, float)
    print("GRPO Objective OK")
    
    # 137 GRPO Adv
    R = [1.0, 2.0, 3.0]
    A_grp = compute_group_relative_advantage(R)
    assert np.allclose(np.mean(A_grp), 0.0)
    print("GRPO Adv OK")
    
    # 138 KL Estimator
    kl = kl_divergence_estimator([1.0], [0.5], [0.5])
    assert np.allclose(kl, 0.0)
    print("KL Estimator OK")

if __name__ == '__main__':
    check()
