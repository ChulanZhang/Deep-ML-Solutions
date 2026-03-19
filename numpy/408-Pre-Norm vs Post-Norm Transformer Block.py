import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, 
                      gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, 
                      mode: str, eps: float = 1e-5) -> np.ndarray:
    """
    Apply a transformer block with two sublayers using either pre-norm or post-norm.
    Assumes Sublayer 1 is a simple linear projection (simulating Attention without QKV complexity for this block) -> Actually just applying W1, b1.
    Wait, let's assume Sublayer1 is: z = x @ W1 + b1
    Sublayer2 is: z = x @ W2 + b2
    """
    if mode == 'pre':
        # Pre-Norm
        # Sublayer 1
        x_norm1 = layer_norm(x, gamma1, beta1, eps)
        sub1_out = np.dot(x_norm1, W1) + b1
        x = x + sub1_out
        
        # Sublayer 2
        x_norm2 = layer_norm(x, gamma2, beta2, eps)
        sub2_out = np.dot(x_norm2, W2) + b2
        x = x + sub2_out
        
    elif mode == 'post':
        # Post-Norm
        # Sublayer 1
        sub1_out = np.dot(x, W1) + b1
        x = layer_norm(x + sub1_out, gamma1, beta1, eps)
        
        # Sublayer 2
        sub2_out = np.dot(x, W2) + b2
        x = layer_norm(x + sub2_out, gamma2, beta2, eps)
        
    return x
