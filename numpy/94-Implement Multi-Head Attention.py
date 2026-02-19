import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model).
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model).
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model).
    """
    # Linear projection: Map the input vectors to Q, K, V subspaces.
    # Note: We use X @ W (shape: (seq_len, d_model) x (d_model, d_model))
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention for a single head.
    Formula: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k) (or d_v)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # 1. Get dimension d_k for scaling
    d_k = K.shape[-1]

    # 2. Compute similarity scores (Scaled Dot Product)
    # Q @ K.T -> (seq_len, seq_len)
    # Divide by sqrt(d_k) to prevent gradients from vanishing/exploding in Softmax
    scaled_dot_product = np.dot(Q, K.T) / np.sqrt(d_k)

    # 3. Apply Softmax to get attention weights
    # Subtract max for numerical stability
    exp_scores = np.exp(scaled_dot_product - np.max(scaled_dot_product, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Compute weighted sum of values
    # (seq_len, seq_len) @ (seq_len, d_k) -> (seq_len, d_k)
    output = np.dot(attention_weights, V)

    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute Multi-Head Attention.
    Instead of performing a single attention function with d_model-dimensional keys, values and queries,
    we find it beneficial to linearly project the queries, keys and values h times with different, learned linear projections.
    
    The mechanism splits the input into multiple 'heads', allows the model to jointly attend to information 
    from different representation subspaces at different positions.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model).
                 These are usually the output of the linear projections (compute_qkv).
        n_heads: Number of attention heads.
    
    Returns:
        Attention output of shape (seq_len, d_model).
    """
    seq_len, d_model = Q.shape
    
    # Validation: d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    
    # Calculate dimension per head
    d_k = d_model // n_heads
    
    # -----------------------------------------------------------
    # Step 1: Split into Heads (Reshape and Transpose)
    # -----------------------------------------------------------
    # We transform Q, K, V from (seq_len, d_model) to (n_heads, seq_len, d_k).
    #
    # Logic:
    # 1. Reshape: (seq_len, d_model) -> (seq_len, n_heads, d_k)
    # 2. Transpose: Swap the first two dimensions to bring n_heads to the front.
    #    New shape: (n_heads, seq_len, d_k).
    #    This allows us to use batch matrix multiplication.
    
    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    # -----------------------------------------------------------
    # Step 2: Scaled Dot-Product Attention (Batched)
    # -----------------------------------------------------------
    # We perform attention for all heads in parallel using matrix operations.
    # 
    # Q_heads shape: (n_heads, seq_len, d_k)
    # K_heads shape: (n_heads, seq_len, d_k)
    #
    # To compute Q @ K^T per head, we transpose the last two dimensions of K_heads.
    # K_heads.transpose(0, 2, 1) -> (n_heads, d_k, seq_len)
    #
    # Resulting Scores shape: (n_heads, seq_len, seq_len)
    
    scores = np.matmul(Q_heads, K_heads.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # -----------------------------------------------------------
    # Step 3: Softmax Normalization
    # -----------------------------------------------------------
    # Apply softmax along the last dimension (key dimension) to get probabilities.
    # We use numerical stability trick (subtract max).
    
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # -----------------------------------------------------------
    # Step 4: Weighted Sum of Values
    # -----------------------------------------------------------
    # Combine values based on attention weights.
    # Weights: (n_heads, seq_len, seq_len)
    # V_heads: (n_heads, seq_len, d_k)
    # Output: (n_heads, seq_len, d_k)
    
    attention_output_heads = np.matmul(attention_weights, V_heads)
    
    # -----------------------------------------------------------
    # Step 5: Concatenate Heads (Merge)
    # -----------------------------------------------------------
    # We need to reshape back to (seq_len, d_model).
    #
    # 1. Transpose back: (n_heads, seq_len, d_k) -> (seq_len, n_heads, d_k)
    #    IMPORTANT: We must reverse the transpose operation done in Step 1 exactly.
    # 2. Reshape: (seq_len, n_heads, d_k) -> (seq_len, n_heads * d_k) -> (seq_len, d_model)
    
    concat_output = attention_output_heads.transpose(1, 0, 2).reshape(seq_len, d_model)
    
    # Typically, there is a final linear projection W_o here.
    # output = np.dot(concat_output, W_o)
    # Since W_o is not provided in args, we return the concatenated output directly.
    
    return concat_output

if __name__ == "__main__":
    # Example Usage
    np.random.seed(42)
    
    # Parameters
    seq_len = 5
    d_model = 16  # Must be divisible by n_heads
    n_heads = 4
    
    # Generate random input X and Weights
    X = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    
    print(f"Input Shape: {X.shape}")
    print(f"Model Dimension: {d_model}")
    print(f"Number of Heads: {n_heads}")
    
    # 1. Compute Q, K, V
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print(f"\ncomputed Q, K, V shape: {Q.shape}")
    
    # 2. Perform Multi-Head Attention
    output = multi_head_attention(Q, K, V, n_heads)
    
    print(f"\nMulti-Head Attention Output Shape: {output.shape}")
    print("Output (first row):\n", output[0])

    # Assert shape correctness
    assert output.shape == (seq_len, d_model), "Output shape mismatch!"
    print("\nTest Passed!")
