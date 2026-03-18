import numpy as np

def learned_positional_encoding(token_embeddings: np.ndarray, position_embedding_table: np.ndarray, start_pos: int = 0) -> np.ndarray:
    """
    Apply learned positional embeddings to token embeddings.
    """
    seq_len = token_embeddings.shape[1]
    positions = position_embedding_table[start_pos:start_pos + seq_len, :]
    return token_embeddings + positions
