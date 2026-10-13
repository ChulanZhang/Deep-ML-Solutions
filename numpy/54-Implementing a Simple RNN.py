import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], 
                Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    """
    Implement a simple RNN cell that processes a sequence.
    
    Formula per step: h_t = tanh(Wx * x_t + Wh * h_{t-1} + b)
    
    Args:
        input_sequence: Sequence of input vectors (sequence_len, input_size)
        initial_hidden_state: Starting hidden state (hidden_size,)
        Wx: Input-to-hidden weights (hidden_size, input_size) - Note: standard ML is often (input, hidden), 
            but we assume dot(Wx, x) or dot(x, Wx) based on shapes. We'll use standard linear algebra.
        Wh: Hidden-to-hidden weights (hidden_size, hidden_size)
        b: Bias (hidden_size,)
        
    Returns:
        Final hidden state rounded to 4 decimal places.
    """
    # Convert lists to numpy arrays for matrix multiplication
    h = np.array(initial_hidden_state, dtype=np.float64)
    Wx = np.array(Wx, dtype=np.float64)
    Wh = np.array(Wh, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    # Iterate through the sequence
    for x_list in input_sequence:
        x = np.array(x_list, dtype=np.float64)
        
        # Depending on how Deep-ML structured Wx and Wh, typically in these problems
        # it is either Wx.dot(x) or x.dot(Wx). 
        # Deep-ML often structures weights such that h = x@Wx + h@Wh + b or h = Wx@x + Wh@h + b.
        # Let's assume standard column-vector matrices where dot(W, x) handles it.
        # If dimensions mismatch, we fall back to dot(x, W).
        try:
            linear_sum = np.dot(Wx, x) + np.dot(Wh, h) + b
        except ValueError:
            linear_sum = np.dot(x, Wx) + np.dot(h, Wh) + b
            
        # Apply tanh activation
        h = np.tanh(linear_sum)
        
    # Round to 4 decimal places inside a standard python list
    return np.round(h, 4).tolist()
