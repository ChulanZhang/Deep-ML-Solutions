import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_k)
        
    Returns:
        Q, K, V: Transformed matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute masked scaled dot-product self-attention.
    Commonly used in Decoder layers (e.g., in Transformers) to prevent attending to future tokens.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        mask: Mask matrix of shape (seq_len, seq_len). 
              Supports two types of masks:
              1. Additive Mask (e.g. -inf for masked, 0 for keep).
              2. Binary Mask (e.g. 0 for masked, 1 for keep).
    
    Returns:
        output: Attention output of shape (seq_len, d_v)
    """
    
    # 1. Get embedding dimension d_k
    # Used for scaling the dot product scores.
    d_k = K.shape[-1]

    # 2. Compute Raw Attention Scores (Dot Product)
    # Calculate similarity between Queries and Keys.
    # Formula: Q @ K^T / sqrt(d_k)
    # Shape: (seq_len, seq_len)
    scaled_dot_product = np.dot(Q, K.T) / np.sqrt(d_k)

    # 3. Apply Masking
    if mask is not None:
        # Check if the mask is an additive mask (contains very small negative numbers like -inf)
        if np.any(mask < -1e4): 
            # Additive mask: Directly add to scores.
            # 0s will not change the score (Keep).
            # -inf will push the score to -inf (Mask).
            scaled_dot_product += mask
        else:
            # Binary mask : Assume 1 = Keep, 0 = Mask
            # Where mask is 0 (invalid positions), we set the score to a very large negative number (-1e9).
            scaled_dot_product = np.where(mask == 0, -1e9, scaled_dot_product)

    # 4. Compute Softmax (Normalization)
    # Convert scores to probabilities.
    # Substracting max for numerical stability (prevents overflow).
    exp_scores = np.exp(scaled_dot_product - np.max(scaled_dot_product, axis=-1, keepdims=True))
    
    # Calculate weights: exp_i / sum(exp_j)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. Weighted Sum of Values
    # Apply the attention weights to the Value matrix V.
    # Output represents the aggregated information for each token, 
    # respecting the mask constraints.
    output = np.dot(attention_weights, V)

    return output

if __name__ == "__main__":
    # Example Usage: Implementation of Causal Masking (Decoder style)
    # We set seed to ensure reproducibility
    np.random.seed(42)
    
    # Parameters
    seq_len = 4
    d_model = 8 
    d_k = 8
    
    # Generate random inputs and weights
    X = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_k)
    
    # Compute Q, K, V
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    
    # Create a Causal Mask (Lower Triangular Matrix)
    # This simulates a scenario where token i can only attend to tokens 0...i
    # Shape: (4, 4)
    # [[1, 0, 0, 0],  <- Token 0 attends only to Token 0
    #  [1, 1, 0, 0],  <- Token 1 attends to Token 0, 1
    #  [1, 1, 1, 0],  ...
    #  [1, 1, 1, 1]]  <- Token 3 attends to all
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    print("Causal Mask:\n", mask)
    
    # Calculate Masked Attention
    output = masked_attention(Q, K, V, mask)
    
    print("\nMasked Attention Output:\n", output)
    print("\nOutput Shape:", output.shape)