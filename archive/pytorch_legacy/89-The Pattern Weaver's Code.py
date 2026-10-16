import torch

def softmax(x):
    """Secure softmax using the max-shift for numerical stability on the last axis."""
    e_x = torch.exp(x - torch.amax(x, dim=-1, keepdim=True))
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)

def calculate_crystal_patterns(number_of_crystals: int, crystal_values: list, dimension: int):
    """
    Simulation of Self-Attention mapped to "The Pattern Weaver's Code" constraint on Deep-ML.
    
    Args:
        number_of_crystals: integer count of items (seq_length)
        crystal_values: 2D array-like mapping to semantic dimension representations
        dimension: Width of each crystal feature vector
    
    Returns:
        New aggregated list array rounded to 3 decimal places representing semantic interactions.
    """
    # 1. Store input array shape: (N, D)
    V = torch.tensor(crystal_values)
    
    # 2. Attention scores: V_i dot V_j
    # This evaluates alignment without W_q and W_k linear layer projections.
    scores = torch.matmul(V, V.T)
    
    # 3. Softmax over alignment weights
    attention_weights = softmax(scores)
    
    # 4. Multiply softmax weights dynamically back against the Value array V
    output = torch.matmul(attention_weights, V)
    
    # Formatting constraint
    return torch.round(output, 3).tolist()
