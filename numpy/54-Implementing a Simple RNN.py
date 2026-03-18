import numpy as np

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> list:
    h = np.array(initial_hidden_state, dtype=float)
    Wx = np.array(Wx, dtype=float)
    Wh = np.array(Wh, dtype=float)
    b = np.array(b, dtype=float)
    
    for x in input_sequence:
        x_val = np.array(x, dtype=float)
        z = np.dot(x_val, Wx) + np.dot(h, Wh) + b
        h = np.tanh(z)
        
    # The output format per the example has rounding 4
    if h.ndim == 1:
        return np.round(h, 4).tolist()
    return np.round(h, 4).tolist()[0] if h.shape[0] == 1 else np.round(h, 4).tolist()
