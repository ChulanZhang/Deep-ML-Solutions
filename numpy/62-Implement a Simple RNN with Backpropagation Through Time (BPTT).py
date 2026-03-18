import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """
        Forward pass for the sequence.
        x: list of vectors (arrays) of shape (input_size, 1) or similar.
        """
        h = {}
        y = {}
        h[-1] = np.zeros((self.W_hh.shape[0], 1))
        
        for t in range(len(x)):
            xt = np.array(x[t]).reshape(-1, 1)
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h[t] = np.tanh(np.dot(self.W_xh, xt) + np.dot(self.W_hh, h[t-1]) + self.b_h)
            # y_t = W_hy * h_t + b_y
            y[t] = np.dot(self.W_hy, h[t]) + self.b_y
            
        return y, h

    def backward(self, x, y_true, learning_rate):
        """
        Backward pass using Backpropagation Through Time (BPTT).
        Loss metric: 1/2 Mean Squared Error.
        """
        y, h = self.forward(x)
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros_like(h[0])
        
        # Traverse time steps backward
        for t in reversed(range(len(x))):
            yt = np.array(y_true[t]).reshape(-1, 1)
            # Derivative of 1/2 (y - y_true)^2 wrapper
            dy = y[t] - yt
            
            # Gradients for Y layer
            dW_hy += np.dot(dy, h[t].T)
            db_y += dy
            
            # Backprop error from y[t] and h[t+1]
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Sub-derivative of tanh
            dtanh = (1 - h[t] ** 2) * dh
            db_h += dtanh
            
            xt = np.array(x[t]).reshape(-1, 1)
            dW_xh += np.dot(dtanh, xt.T)
            dW_hh += np.dot(dtanh, h[t-1].T)
            
            # Update next delta h for previous sequence position
            dh_next = np.dot(self.W_hh.T, dtanh)
            
        # Optional safe-guard: Gradient clipping to avoid explosions in RNNs
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)
            
        # Update weights and biases
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
        
        return y, h
