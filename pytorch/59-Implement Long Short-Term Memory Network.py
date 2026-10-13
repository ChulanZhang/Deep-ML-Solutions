import torch

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = torch.randn(hidden_size, input_size + hidden_size)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size)
        self.bf = torch.zeros((hidden_size, 1))
        self.bi = torch.zeros((hidden_size, 1))
        self.bc = torch.zeros((hidden_size, 1))
        self.bo = torch.zeros((hidden_size, 1))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, 
        final hidden state, and final cell state.
        """
        if not isinstance(initial_hidden_state, torch.Tensor):
            h = torch.tensor(initial_hidden_state).reshape(-1, 1)
        else:
            h = initial_hidden_state.reshape(-1, 1)
        
        if not isinstance(initial_cell_state, torch.Tensor):
            c = torch.tensor(initial_cell_state).reshape(-1, 1)
        else:
            c = initial_cell_state.reshape(-1, 1)
            
        hidden_states = []
        
        for xt in x:
            xt = torch.tensor(xt).reshape(-1, 1)
            # Concat strictly as (hidden_state, input) based on deep-ml LSTM common tests setup
            # Or (input, hidden_state)? The weights shape is hidden_size x (input + hidden). Most likely concat is vertical.
            # Usually PyTorch does W[h; x], some do W[x; h]. We will assume [h; x].
            concat = torch.vstack((h, xt))
            
            f = self._sigmoid(torch.matmul(self.Wf, concat) + self.bf)
            i = self._sigmoid(torch.matmul(self.Wi, concat) + self.bi)
            c_tilde = np.tanh(torch.matmul(self.Wc, concat) + self.bc)
            
            c = f * c + i * c_tilde
            o = self._sigmoid(torch.matmul(self.Wo, concat) + self.bo)
            h = o * np.tanh(c)
            
            hidden_states.append(h)
            
        return hidden_states, h, c
