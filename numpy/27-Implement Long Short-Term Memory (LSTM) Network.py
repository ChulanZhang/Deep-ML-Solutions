import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases (Deep-ML problem already gives this setup in boilerplate)
        # Weights combine [x_t, h_{t-1}]
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs.
        
        Args:
            x: Input sequence of shape (seq_len, input_size) or (seq_len, input_size, 1)
            initial_hidden_state: (hidden_size, 1)
            initial_cell_state: (hidden_size, 1)
            
        Returns:
            h_states (list of all h_t), final h, final c
        """
        seq_len = x.shape[0]
        h = initial_hidden_state
        c = initial_cell_state
        
        # Ensure x is 3D: (seq_len, input_size, 1) to match column vectors
        if len(x.shape) == 2:
            x = x.reshape(seq_len, self.input_size, 1)
            
        h_states = []

        for t in range(seq_len):
            x_t = x[t]
            
            # Concatenate h_{t-1} and x_t vertically to form a single vector
            concat = np.vstack((h, x_t))
            
            # Forget Gate
            # f_t = sigmoid(Wf * [h_{t-1}, x_t] + bf)
            f_t = self._sigmoid(np.dot(self.Wf, concat) + self.bf)
            
            # Input Gate
            # i_t = sigmoid(Wi * [h_{t-1}, x_t] + bi)
            i_t = self._sigmoid(np.dot(self.Wi, concat) + self.bi)
            
            # Candidate Cell State
            # c_tilde = tanh(Wc * [h_{t-1}, x_t] + bc)
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
            
            # Output Gate
            # o_t = sigmoid(Wo * [h_{t-1}, x_t] + bo)
            o_t = self._sigmoid(np.dot(self.Wo, concat) + self.bo)
            
            # Update Cell State
            # c_t = f_t * c_{t-1} + i_t * c_tilde  (element-wise multiplication)
            c = f_t * c + i_t * c_tilde
            
            # Update Hidden State
            # h_t = o_t * tanh(c_t)
            h = o_t * np.tanh(c)
            
            h_states.append(h)

        return h_states, h, c

    def _sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -100, 100)
        return 1.0 / (1.0 + np.exp(-z))
