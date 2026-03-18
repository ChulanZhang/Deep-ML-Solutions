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

mod39 = load_module("m39", "39-Implementation of Log Softmax Function.py")
mod42 = load_module("m42", "42-Implement ReLU Activation Function.py")
mod44 = load_module("m44", "44-Leaky ReLU Activation Function.py")
mod49 = load_module("m49", "49-Implement Adam Optimization Algorithm.py")
mod54 = load_module("m54", "54-Implementing a Simple RNN.py")
mod59 = load_module("m59", "59-Implement Long Short-Term Memory Network.py")
mod70 = load_module("m70", "70-Calculate Image Brightness.py")
mod82 = load_module("m82", "82-Grayscale Image Contrast Calculator.py")
mod146 = load_module("m146", "146-Momentum Optimizer.py")
mod147 = load_module("m147", "147-GeLU Activation Function.py")
mod148 = load_module("m148", "148-Adamax Optimizer.py")
mod149 = load_module("m149", "149-Adadelta Optimizer.py")
mod151 = load_module("m151", "151-Dropout Layer.py")
mod156 = load_module("m156", "156-Implement SwiGLU activation function.py")
mod157 = load_module("m157", "157-Implement the Bellman Equation for Value Iteration.py")

def check():
    # 39
    assert np.allclose(mod39.log_softmax([1, 2, 3]), [-2.4076, -1.4076, -0.4076], atol=1e-3)
    # 42
    assert mod42.relu(-1) == 0.0
    assert mod42.relu(1) == 1.0
    # 44
    assert mod44.leaky_relu(-2, 0.1) == -0.2
    # 49
    x0 = [1.0, 1.0]
    opt = mod49.adam_optimizer(None, lambda x: np.array([2*x[0], 2*x[1]]), x0, num_iterations=10)
    assert len(opt) == 2
    # 54
    inp = [[1.0], [2.0], [3.0]]
    rnn_out = mod54.rnn_forward(inp, [0.0], [[0.5]], [[0.8]], [0.0])
    assert rnn_out == [0.9759] or np.allclose(rnn_out, [0.9759])
    # 59
    lstm = mod59.LSTM(input_size=2, hidden_size=3)
    x_lstm = np.random.randn(5, 2)
    h_init = np.zeros((3, 1))
    c_init = np.zeros((3, 1))
    hs, h, c = lstm.forward(x_lstm, h_init, c_init)
    assert len(hs) == 5
    # 70
    assert mod70.calculate_brightness([[100, 200], [50, 150]]) == 125.0
    # 82
    assert mod82.calculate_contrast([[0, 50], [200, 255]]) == 255.0
    # 146
    p, v = mod146.momentum_optimizer(1.0, 0.1, 0.1)
    assert np.allclose([p, v], [0.909, 0.091])
    # 147
    gelu = mod147.GeLU([-2.0, 0.0, 2.0])
    assert len(gelu) == 3
    # 148
    pa, ma, ua = mod148.adamax_optimizer(1.0, 0.1, 0.0, 0.0, 1)
    # 149
    pd, ud, vd = mod149.adadelta_optimizer(1.0, 0.1, 1.0, 1.0)
    # 151
    drop = mod151.DropoutLayer(0.5)
    do = drop.forward(np.array([1., 2., 3., 4.]))
    assert len(do) == 4
    # 156
    swi = mod156.SwiGLU([[1, -1, 1000, -1000]])
    # 157
    trans = {0: {0: [(1.0, 0, 1.0)]}, 1: {0: [(1.0, 1, 1.0)]}}
    V_new = mod157.bellman_update([0.0, 0.0], trans, 0.9)
    assert np.allclose(V_new, [1.0, 1.0])
    
    print("Batch 13 OK")

if __name__ == '__main__':
    check()
