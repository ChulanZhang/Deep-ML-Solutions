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

mod139 = load_module("lora", "139-LoRA Forward Pass.py")
mod140 = load_module("qlora", "140-QLoRA Forward Pass.py")
mod141 = load_module("ppl", "141-Calculate Perplexity.py")
mod142 = load_module("cb", "142-Continuous Batching Simulator.py")
mod143 = load_module("fa", "143-Flash Attention Forward.py")
mod144 = load_module("roof", "144-Roofline Model Analysis.py")
mod145 = load_module("qc", "145-Quantization Quality Check.py")

lora_forward = mod139.lora_forward
qlora_forward = mod140.qlora_forward
calculate_perplexity = mod141.calculate_perplexity
continuous_batching_sim = mod142.continuous_batching_sim
flash_attention_forward = mod143.flash_attention_forward
roofline_analysis = mod144.roofline_analysis
quantization_quality_check = mod145.quantization_quality_check

def check():
    # 139 LoRA
    x = [[1.0, 2.0]]
    W = [[1.0], [0.5]]
    A = [[0.1]]
    B = [[0.2], [0.3]]
    out = lora_forward(x, W, A, B, 1.0)
    assert len(out) == 1 and len(out[0]) == 1
    print("LoRA OK")
    
    # 140 QLoRA
    qw = [[1], [0]]
    outq = qlora_forward(x, qw, 0.5, 0.0, A, B, 1.0)
    assert len(outq) == 1
    print("QLoRA OK")
    
    # 141 Perplexity
    ppl = calculate_perplexity([0.5, 0.5])
    assert ppl > 1.0
    print("Perplexity OK")
    
    # 142 Continuous Batching
    reqs = [{'arrival_time': 0, 'tokens_needed': 2}, {'arrival_time': 0, 'tokens_needed': 1}]
    res_cb = continuous_batching_sim(reqs, 1)
    # req0 finishes at 2. req1 starts at 2, finishes at 3. time=3.
    assert res_cb['total_time'] == 3
    print("Continuous Batching OK")
    
    # 143 FlashAttn
    Q = np.ones((4, 2))
    K = np.ones((4, 2))
    V = np.ones((4, 2))
    fa = flash_attention_forward(Q, K, V, 2)
    assert fa.shape == (4, 2)
    print("FlashAttention OK")
    
    # 144 Roofline
    ops = [{'name': 'op1', 'flops': 100, 'bytes': 10}]
    r = roofline_analysis(100.0, 50.0, ops)
    assert r['ridge_point'] == 2.0
    assert r['operations'][0]['bottleneck'] == 'compute-bound'
    print("Roofline OK")
    
    # 145 Quant Check
    orig = [-0.1, -0.2]
    quant = [-0.15, -0.25]
    qc = quantization_quality_check(orig, quant)
    assert qc['quality'] in ['excellent', 'acceptable', 'poor']
    print("Quant Quality Check OK")

if __name__ == '__main__':
    check()
