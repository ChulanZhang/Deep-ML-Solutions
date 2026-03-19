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

mod421 = load_module("m421", "421-GPU Ops_Byte Ratio Calculation from Spec Sheet.py")
mod422 = load_module("m422", "422-Multi-GPU Communication Overhead (NVLink vs InfiniBand).py")
mod423 = load_module("m423", "423-Multi-Instance GPU (MIG) Resource Allocation.py")
mod424 = load_module("m424", "424-Kernel Fusion Memory Savings Calculator.py")
mod425 = load_module("m425", "425-Continuous Batching (In-Flight Batching) Simulator.py")
mod426 = load_module("m426", "426-Post-Training Quantization with Per-Channel Scale Factors.py")
mod427 = load_module("m427", "427-FP4 Quantization with Microscaling (MXFP4).py")
mod428 = load_module("m428", "428-Number Format Precision Comparison (FP16 vs BF16 vs FP8 vs FP4).py")
mod431 = load_module("m431", "431-EAGLE-Style Draft Model from Hidden States.py")
mod435 = load_module("m435", "435-KV Cache Memory Budget and Eviction Policy.py")
mod438 = load_module("m438", "438-Tensor Parallelism All-Reduce Communication Cost.py")
mod442 = load_module("m442", "442-VLM Visual Token Count from Image Resolution and Patch Size.py")
mod446 = load_module("m446", "446-Video Generation Latent Space Memory Estimation.py")
mod447 = load_module("m447", "447-Classifier-Free Guidance Skip Speedup Calculator.py")
mod448 = load_module("m448", "448-Context Parallelism with Ring Attention for Video Models.py")

def check():
    r421 = mod421.gpu_ops_byte_ratio(312.0, 1500.0)
    assert isinstance(r421, float)
    
    r422 = mod422.multi_gpu_communication_overhead(100.0, 300.0, 8, 2.0)
    assert isinstance(r422, float)
    
    sm, mem = mod423.mig_resource_allocation(108, 80.0, 7)
    assert isinstance(sm, int) and isinstance(mem, float)
    
    s_mb = mod424.kernel_fusion_savings(1, 10, 512, 3)
    assert isinstance(s_mb, float)
    
    assert hasattr(mod425, 'continuous_batching_sim') or hasattr(mod425, 'simulate_continuous_batching')
    
    q, s = mod426.per_channel_quantization(np.random.randn(4, 10))
    assert q.shape == (4, 10)
    
    assert hasattr(mod427, 'mxfp4_quantize')
    
    cp = mod428.float_precision_comparison([1.12345], 'fp16')
    assert len(cp) == 1
    
    eg = mod431.eagle_draft_model([[1, 2]], [[0.5, -0.5], [-0.5, 0.5]])
    assert len(eg) == 1
    
    kv = mod435.kv_cache_eviction([1, 2, 3], [4, 5], 4)
    assert len(kv) == 4
    
    tp = mod438.tensor_parallel_allreduce(100.0, 300.0, 8)
    assert isinstance(tp, float)
    
    # 442
    v_toks = mod442.vlm_visual_tokens(224, 224, 16)
    assert v_toks == 196
    
    # 446
    v_mem = mod446.video_latent_memory(16, 4, 64, 64)
    assert isinstance(v_mem, float)
    
    # 447
    sp = mod447.cfg_skip_speedup(20, 0.5)
    assert isinstance(sp, float)
    
    # 448
    ra = mod448.context_parallelism_ring(1024, 8, 128)
    assert ra == 7
    
    print("Batch 21 OK")

if __name__ == '__main__':
    check()
