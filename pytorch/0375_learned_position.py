"""Deep-ML #375: Learned Positional Embeddings.
https://www.deep-ml.com/problems/375
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def learned_positional_encoding(token_embeddings, position_embedding_table, start_pos=0):
    length = token_embeddings.shape[-2]
    if start_pos < 0 or start_pos + length > position_embedding_table.shape[0]:
        raise ValueError('Positions are outside the embedding table.')
    return token_embeddings + position_embedding_table[start_pos:start_pos + length]
