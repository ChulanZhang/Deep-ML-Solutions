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

mod121 = load_module("ffn", "121-Implement Position-wise Feed-Forward Block with Residual and Dropout.py")
mod122 = load_module("srelu", "122-Implement the Square ReLU Activation Function.py")
mod123 = load_module("linlr", "123-Linear Learning Rate Decay.py")
mod124 = load_module("cntparam", "124-Calculate Number of Parameters in Neural Network.py")
mod125 = load_module("digit", "125-Build a Digit Classifier from Scratch.py")
mod126 = load_module("temp", "126-Temperature Sampling.py")

ffn = mod121.ffn
square_relu = mod122.square_relu
linear_lr_decay = mod123.linear_lr_decay
count_parameters = mod124.count_parameters
train_digit = mod125.train
temperature_sampling = mod126.temperature_sampling

def check():
    # 121 FFN
    x = np.ones((2, 3))
    w1 = np.ones((3, 4))
    b1 = np.zeros(4)
    w2 = np.ones((4, 3))
    b2 = np.zeros(3)
    out = ffn(x, w1, b1, w2, b2, 0.5, 42)
    assert out.shape == (2, 3)
    print("FFN OK")
    
    # 122 Square ReLU
    x_s = np.array([-1.0, 0.0, 2.0])
    res = square_relu(x_s)
    assert np.allclose(res['output'], [0.0, 0.0, 4.0])
    assert np.allclose(res['derivative'], [0.0, 0.0, 4.0])
    print("Square ReLU OK")
    
    # 123 linear LR decay
    schedule = linear_lr_decay(0.1, 0.0, 3)
    assert np.allclose(schedule, [0.1, 0.05, 0.0])
    print("Linear LR Decay OK")
    
    # 124 count parameters
    layers = [{'type': 'dense', 'input_dim': 2, 'output_dim': 3, 'bias': True}]
    assert count_parameters(layers) == 9
    print("Count Parameters OK")
    
    # 125 digit classifier
    X_t = [[0.5]*64, [0.1]*64]
    y_t = [0, 1]
    predict = train_digit(X_t, y_t, X_t, y_t, 2)
    preds = predict(X_t)
    assert len(preds) == 2
    print("Digit Classifier OK")
    
    # 126 temperature sampling
    logits = [1.0, 2.0, 3.0]
    probs = temperature_sampling(logits, 1.0)
    assert len(probs) == 3
    probs_zero = temperature_sampling(logits, 0.0)
    assert probs_zero == [0.0, 0.0, 1.0]
    print("Temperature Sampling OK")

if __name__ == '__main__':
    check()
