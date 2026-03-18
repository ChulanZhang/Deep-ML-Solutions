import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, 
        final hidden state, and final cell state.
        """
        if not isinstance(initial_hidden_state, np.ndarray):
            h = np.array(initial_hidden_state).reshape(-1, 1)
        else:
            h = initial_hidden_state.reshape(-1, 1)
        
        if not isinstance(initial_cell_state, np.ndarray):
            c = np.array(initial_cell_state).reshape(-1, 1)
        else:
            c = initial_cell_state.reshape(-1, 1)
            
        hidden_states = []
        
        for xt in x:
            xt = np.array(xt).reshape(-1, 1)
            # Concat strictly as (hidden_state, input) based on deep-ml LSTM common tests setup
            # Or (input, hidden_state)? The weights shape is hidden_size x (input + hidden). Most likely concat is vertical.
            # Usually PyTorch does W[h; x], some do W[x; h]. We will assume [h; x].
            concat = np.vstack((h, xt))
            
            f = self._sigmoid(np.dot(self.Wf, concat) + self.bf)
            i = self._sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
            
            c = f * c + i * c_tilde
            o = self._sigmoid(np.dot(self.Wo, concat) + self.bo)
            h = o * np.tanh(c)
            
            hidden_states.append(h)
            
        return hidden_states, h, c
