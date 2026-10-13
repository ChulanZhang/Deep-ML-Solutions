import torch

def learned_positional_encoding(token_embeddings: torch.Tensor, position_embedding_table: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Apply learned positional embeddings to token embeddings.
    """
    seq_len = token_embeddings.shape[1]
    positions = position_embedding_table[start_pos:start_pos + seq_len, :]
    return token_embeddings + positions
