"""
Principles of Layer Normalization (LN)

1. Core Concept:
   Normalize all features within a single sample. Unlike Batch Normalization (BN), 
   LN does not depend on the batch size, making it ideal for variable-length sequence data.

2. Mathematical Formula:
   For an input vector x (with dimension H):
   a. Compute Mean μ: μ = (1/H) * Σ(x_i)
   b. Compute Variance σ²: σ² = (1/H) * Σ(x_i - μ)²
   c. Normalization: x_hat = (x - μ) / sqrt(σ² + ε)
   d. Scale and Shift: y = γ * x_hat + β
      Where γ (gamma) and β (beta) are learnable parameters.

3. LN vs Batch Normalization (BN):
   - BN (Vertical Normalization): Operates along the batch dimension, computing statistics 
     for the same feature across different samples. Highly dependent on batch size.
   - LN (Horizontal Normalization): Operates along the feature dimension, computing 
     statistics across different features within the same sample. Independent of batch size.

4. Why Transformer Prefers LN?
   - Handling Variable Lengths: Each token is normalized independently, unaffected by sequence length.
   - Training Stability: Effectively mitigates vanishing/exploding gradients and 
     accelerates model convergence.
   - Simplified Inference: No need to maintain global running means/variances during inference.
"""

import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization on an input tensor X.
    
    Layer Normalization normalizes the input across the feature dimension (last axis).
    For a 3D input (batch_size, seq_len, d_model), it computes mean and variance 
    for each vector of size d_model.
    
    Formula:
        Output = gamma * (X - mean) / sqrt(var + epsilon) + beta
    
    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        gamma: Scaling parameter of shape (d_model,)
        beta: Shifting parameter of shape (d_model,)
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized tensor of the same shape as X.
    """
    
    # 1. Calculate Mean (along the last dimension)
    # Shape: (batch_size, seq_len, 1)
    mean = np.mean(X, axis=-1, keepdims=True)
    
    # 2. Calculate Variance (along the last dimension)
    # Shape: (batch_size, seq_len, 1)
    var = np.var(X, axis=-1, keepdims=True)
    
    # 3. Normalize
    # We subtract the mean and divide by the standard deviation.
    # epsilon is added inside the square root to prevent division by zero.
    X_hat = (X - mean) / np.sqrt(var + epsilon)
    
    # 4. Scale and Shift (Affine Transformation)
    # gamma and beta are broadcated across (batch_size, seq_len) dimensions.
    output = gamma * X_hat + beta
    
    return output

if __name__ == "__main__":
    # Example Usage:
    # -----------------------------------------------------------
    np.random.seed(42)
    
    batch_size = 2
    seq_len = 3
    d_model = 4
    
    # Generate random input
    X = np.random.randn(batch_size, seq_len, d_model)
    
    # Initialize gamma (scale) to 1s and beta (bias) to 0s
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    
    print("Input X (first batch, first sequence):\n", X[0, 0])
    
    # Apply Layer Normalization
    output = layer_normalization(X, gamma, beta)
    
    print("\nLayer Normalized Output (first batch, first sequence):\n", output[0, 0])
    
    # Verify: Mean should be ~0 and Variance should be ~1 across the feature dimension
    print("\nMean (after LN) across feature dim [0,0]:", np.mean(output[0, 0]))
    print("Var (after LN) across feature dim [0,0]:", np.var(output[0, 0]))
    
    # Test with custom gamma and beta
    gamma_custom = np.array([0.5, 2.0, 1.0, 0.1])
    beta_custom = np.array([10.0, -5.0, 0.0, 1.0])
    output_custom = layer_normalization(X, gamma_custom, beta_custom)
    
    print("\nOutput with custom Gamma/Beta (first row):\n", output_custom[0, 0])