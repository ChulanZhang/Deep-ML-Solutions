import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    This is the first step of the self-attention mechanism, projecting input vectors into different subspaces.

    Args:
        X: Input matrix.
           Shape is typically (seq_len, input_dim) or (batch_size, seq_len, input_dim).
           Here we assume 2D (seq_len, input_dim).
        W_q: Query weight matrix, shape (input_dim, d_k).
        W_k: Key weight matrix, shape (input_dim, d_k).
        W_v: Value weight matrix, shape (input_dim, d_v).

    Returns:
        Q, K, V: The Query, Key, and Value matrices respectively.
    """
    # Linear transformations: project input X into Q, K, V spaces via matrix multiplication.
    # np.dot(a, b) operates as matrix multiplication for 2D arrays.
    Q = np.dot(X, W_q)  # Shape: (seq_len, d_k)
    K = np.dot(X, W_k)  # Shape: (seq_len, d_k)
    V = np.dot(X, W_v)  # Shape: (seq_len, d_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    Core formula: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    
    Args:
        Q: Query matrix, shape (seq_len, d_k).
           Represents "what we are looking for".
        K: Key matrix, shape (seq_len, d_k).
           Represents "what can be queried".
        V: Value matrix, shape (seq_len, d_v).
           Represents "the actual content/information".
    
    Returns:
        output: The output of the attention mechanism, shape (seq_len, d_v).
           Review contextualized vector representations.
    """
    
    # 1. Get the dimension d_k of key vectors.
    # K.shape[-1] retrieves the size of the last dimension, which is the feature dimension d_k.
    d_k = K.shape[-1]

    # 2. Compute Attention Scores.
    # Use dot product to measure similarity between Query and Key.
    # Q @ K.T
    # Q: (seq_len, d_k)
    # K.T: (d_k, seq_len)
    # Result shape: (seq_len, seq_len) -> represents correlation scores between each word and every other word in the sequence.
    # 
    # Why divide by sqrt(d_k)?
    # This is the "Scaling" operation. When d_k is large, the dot product results can be very large,
    # pushing the Softmax function into regions with extremely small gradients (saturation zone), making training difficult.
    # Dividing by sqrt(d_k) stabilizes the variance around 1.
    scaled_dot_product = np.dot(Q, K.T) / np.sqrt(d_k)

    # 3. Compute Softmax Normalization.
    # Convert scores into a probability distribution (weights) so that weights output to 1.
    # Numerical stability trick: subtract the max value (x - max(x)) to prevent exp(x) overflow.
    # axis=-1, keepdims=True ensures the operation is performed across each row (all Keys for a specific Query).
    exp_scores = np.exp(scaled_dot_product - np.max(scaled_dot_product, axis=-1, keepdims=True))
    
    # Calculate attention weights.
    # weights = exp_scores / sum(exp_scores)
    # attention_weights shape: (seq_len, seq_len), where each row sums to 1.
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Compute Weighted Sum to get Output.
    # Apply attention weights to the Value matrix.
    # This aggregates information from all Values based on their relevance (weights).
    # (seq_len, seq_len) @ (seq_len, d_v) -> (seq_len, d_v)
    output = np.dot(attention_weights, V)

    return output 

if __name__ == "__main__":
    # Example Usage
    np.random.seed(42)
    
    # Assume input sequence length is 3, input dimension is 4 (seq_len=3, input_dim=4)
    seq_len = 3
    input_dim = 4
    d_k = 2  # Key/Query dimension
    d_v = 3  # Value dimension

    X = np.random.randn(seq_len, input_dim)
    W_q = np.random.randn(input_dim, d_k)
    W_k = np.random.randn(input_dim, d_k)
    W_v = np.random.randn(input_dim, d_v)

    print("Input X shape:", X.shape)
    
    # Calculate Q, K, V
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    
    print("Computed Q shape:", Q.shape)
    print("Computed K shape:", K.shape)
    print("Computed V shape:", V.shape)

    # Perform Self-Attention
    output = self_attention(Q, K, V)
    print("\nAttention Output:\n", output)
    print("\nOutput Shape:", output.shape)
