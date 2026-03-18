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

mod289 = load_module("m289", "289-Implement Xavier_Glorot Weight Initialization.py")
mod290 = load_module("m290", "290-Implement He Weight Initialization.py")
mod291 = load_module("m291", "291-Calculate Number of Parameters in Neural Network.py")
mod292 = load_module("m292", "292-Implement Gradient Clipping by Value.py")
mod294 = load_module("m294", "294-Implement INT8 Quantization.py")
mod298 = load_module("m298", "298-Implement mHC Forward Pass.py")
mod302 = load_module("m302", "302-Diffusion Reconstruction Loss.py")
mod303 = load_module("m303", "303-Forward Diffusion Process.py")
mod304 = load_module("m304", "304-Forward & Backward Diffusion Process.py")
mod316 = load_module("m316", "316-MMLU Log-Probability Scoring.py")
mod317 = load_module("m317", "317-Rubric-Based LLM Judge Evaluation.py")
mod318 = load_module("m318", "318-Boxed Answer Extraction for Math Benchmarks.py")
mod319 = load_module("m319", "319-Math Answer Verification with Equivalence Checking.py")
mod323 = load_module("m323", "323-Pairwise Preference Judge for LLM Comparison.py")
mod324 = load_module("m324", "324-Code Execution Verifier for Programming Benchmarks.py")

def check():
    r289 = mod289.xavier_initialization((3, 4))
    r290 = mod290.he_initialization((3, 4))
    r291 = mod291.count_parameters([{'in': 2, 'out': 3, 'bias': True}])
    assert r291 == 9
    r292 = mod292.clip_gradients_value([[10.0, -10.0]], 5.0)
    assert np.all(r292[0] == [5.0, -5.0])
    
    r294, s = mod294.int8_quantize(np.array([[2.0, -2.0]]))
    
    Q = np.random.randn(2, 5, 8)
    K = np.random.randn(2, 6, 8)
    V = np.random.randn(2, 6, 8)
    r298 = mod298.mhc_forward(Q, K, V, 2)
    assert r298.shape == (2, 5, 8)
    
    r302 = mod302.diffusion_reconstruction_loss([0.5, 0.5], [0.5, 0.5])
    assert r302 == 0.0
    r303 = mod303.forward_diffusion([1.0], 1, [0.9, 0.8], [0.1])
    assert len(r303) == 1
    r304f, r304b = mod304.diffusion_processes([1.0], 1, [0.1, 0.9], [0.9, 0.8], [0.1], [0.1], [0.1])
    
    r316 = mod316.mmlu_scoring({'A': -0.1, 'B': -2.0})
    assert r316 == 'A'
    
    r317 = mod317.rubric_judge("The sky is blue", ["sky"])
    assert r317 == 1
    
    r318 = mod318.extract_boxed_answer("Answer is \\boxed{42} strictly.")
    assert r318 == "42"
    
    r319 = mod319.verify_math_answer("42.0", "42")
    assert r319 == True
    
    r323 = mod323.pairwise_judge("msg", "good", "bad", lambda p, r: len(r))
    assert r323 == "A"
    
    code = "def f(x): return x * 2"
    r324 = mod324.verify_code(code, [([2], 4)])
    assert r324 == 1.0
    
    print("Batch 17 OK")

if __name__ == '__main__':
    check()
