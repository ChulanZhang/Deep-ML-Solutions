def train_tokenizer(corpus, vocab_size):
    """
    Build a simple character-level tokenizer from the training corpus.
    """
    text = "".join(corpus)
    unique_chars = sorted(list(set(text)))
    
    stoi = {ch: i for i, ch in enumerate(unique_chars[:vocab_size])}
    itos = {i: ch for i, ch in enumerate(unique_chars[:vocab_size])}
    
    def encode(text_in: str) -> list:
        return [stoi[ch] for ch in text_in if ch in stoi]
        
    def decode(indices: list) -> str:
        return "".join(itos[i] for i in indices if i in itos)
        
    return encode, decode
