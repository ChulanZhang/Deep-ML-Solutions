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

mod402 = load_module("m402", "402-Latent Diffusion Encoding and Decoding.py")
mod403 = load_module("m403", "403-Diffusion Cosine Noise Schedule.py")
mod404 = load_module("m404", "404-Implement Score Matching for Score-Based Diffusion.py")
mod406 = load_module("m406", "406-NoPE (No Positional Embedding) with iRoPE Attention.py")
mod407 = load_module("m407", "407-QK-Norm (Query-Key Normalization).py")
mod408 = load_module("m408", "408-Pre-Norm vs Post-Norm Transformer Block.py")
mod409 = load_module("m409", "409-MoE with Shared Expert Forward Pass.py")
mod410 = load_module("m410", "410-Speculative Decoding End-to-End Simulation.py")
mod414 = load_module("m414", "414-Compute Arithmetic Intensity and Classify Bottleneck.py")
mod415 = load_module("m415", "415-Roofline Model Analysis for GPU Operations.py")
mod416 = load_module("m416", "416-Compute Attention Memory Traffic and FLOPs.py")
mod417 = load_module("m417", "417-Classify LLM Prefill vs Decode as Compute-Bound or Memory-Bound.py")
mod418 = load_module("m418", "418-Estimate KV Cache Size from Model Config.py")
mod419 = load_module("m419", "419-Combined Token Sampling Pipeline (Temperature + Top-k + Top-p).py")
mod420 = load_module("m420", "420-Latent Space Patchification for Diffusion Transformers.py")

def check():
    lat, re = mod402.latent_diffusion_codec(np.random.randn(2, 4), np.random.randn(4, 2), np.random.randn(2, 4))
    assert lat.shape == (2, 2)
    assert re.shape == (2, 4)
    
    b, a, ab = mod403.cosine_beta_schedule(10)
    assert len(b) == 10
    
    loss = mod404.score_matching_loss([0], [1], [0], 1.0)
    assert isinstance(loss, float)
    
    assert hasattr(mod406, 'apply_rope')
    
    Q, K = mod407.qk_norm(np.random.randn(2, 4), np.random.randn(2, 4))
    assert Q.shape == (2, 4)
    
    assert hasattr(mod408, 'transformer_block')
    
    out = mod409.shared_expert_moe(np.random.randn(2, 4), np.random.randn(4, 4), [np.random.randn(4, 4)], np.random.randn(4, 1))
    assert out.shape == (2, 4)
    
    s_time, n_time = mod410.speculative_decoding_sim(100, 0.8, 10.0, 50.0)
    assert isinstance(s_time, float)
    
    ai, bot = mod414.arithmetic_intensity(1000, 10)
    assert bot in ["Compute-Bound", "Memory-Bound"]
    
    assert hasattr(mod415, 'roofline_analysis')
    
    mem, flo = mod416.compute_attention_memory_traffic(1, 10, 8, 32)
    assert isinstance(mem, int)
    
    bot = mod417.classify_llm_phase(1, 10, 1_000_000, 1e12, 1e9)
    assert bot in ["Compute-Bound", "Memory-Bound"]
    
    sz = mod418.estimate_kv_cache_size(1, 10, 12, 8, 32)
    assert isinstance(sz, float)
    
    p = mod419.token_sampling_pipeline([1.0, 2.0, 3.0, 4.0], 1.0, 2, 0.9)
    assert len(p) == 4
    
    patch = mod420.latent_diff_patchify(np.random.randn(3, 8, 8), 2)
    assert patch.shape == (16, 12)
    
    print("Batch 20 OK")

if __name__ == '__main__':
    check()
