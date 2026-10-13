class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        """
        unique_chars = sorted(list(set(text)))
        self.stoi = {'<BOS>': 0, '<EOS>': 1}
        self.itos = {0: '<BOS>', 1: '<EOS>'}
        
        for i, char in enumerate(unique_chars):
            idx = i + 2
            self.stoi[char] = idx
            self.itos[idx] = char
            
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        """
        indices = [self.stoi['<BOS>']]
        for char in text:
            # Assumes char is in vocab
            indices.append(self.stoi[char])
        indices.append(self.stoi['<EOS>'])
        return indices

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        """
        chars = []
        for idx in indices:
            chars.append(self.itos[idx])
        return "".join(chars)
