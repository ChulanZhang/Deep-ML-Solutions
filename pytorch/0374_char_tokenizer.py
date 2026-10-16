"""Deep-ML #374. https://www.deep-ml.com/problems/374
Public examples redefine the class themselves; independent tests validate this implementation.
The exercise decode includes literal <BOS>/<EOS> tokens.
"""
import torch
import torch.nn as nn

class CharTokenizer:
    def __init__(self, text: str):
        vocabulary = ['<BOS>', '<EOS>'] + sorted(set(text))
        self.stoi = {token: i for i, token in enumerate(vocabulary)}
        self.itos = {i: token for token, i in self.stoi.items()}
        self.vocab_size = len(vocabulary)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)

    def encode(self, text: str) -> torch.Tensor:
        ids = [0] + [self.stoi[ch] for ch in text] + [1]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, indices) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join(self.itos[int(i)] for i in indices)
