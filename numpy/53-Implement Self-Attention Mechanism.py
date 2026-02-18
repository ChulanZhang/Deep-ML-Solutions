import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k = K.shape[-1]

    scaled_dot_product = np.dot(Q, K.T) / np.sqrt(d_k)

    exp_scores = np.exp(scaled_dot_product - np.max(scaled_dot_product, axis=-1, keepdims=True))

    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = np.dot(attention_weights, V)

    return output 
