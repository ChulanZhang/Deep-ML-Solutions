import numpy as np

"""
Principles of Positional Encoding (位置编码)

1. Why do we need it? (为什么需要它？)
   Transformers use self-attention, which does not inherently account for the order of tokens in a sequence. 
   To provide the model with "positional awareness," we must inject information about the relative or 
   absolute position of each token.
   Transformer 采用自注意力机制，本身不具备感知序列顺序的能力。为了让模型识别 token 的相对或绝对位置信息，
   我们需要引入“位置编码”。

2. Mathematical Formula (数学公式):
   For a given position 'pos' and dimension index 'i':
   - PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
   
   Here, 'd_model' is the total dimensionality of the embedding.

3. Key Characteristics (核心特点):
   - Sine & Cosine: The alternating sine/cosine pattern allows the model to easily learn 
     to attend by relative positions, because for any fixed offset 'k', PE(pos+k) can be 
     represented as a linear function of PE(pos).
   - Scaling: The 10000^ exponent creates a geometric progression of wavelengths, 
     allowing the model to handle long sequences.
   - 交替正余弦：这种设计允许模型学习相对位置关系，因为 PE(pos+k) 可以表示为 PE(pos) 的线性组合。
   - 几何级数波长：指数项确保了不同维度具有不同的波长，从而覆盖长短不一的依赖关系。
"""

def pos_encoding(position: int, d_model: int):
    """
    Calculate positional encodings for a sequence.
    
    Args:
        position: The sequence length (total number of positions to encode).
        d_model: The dimensionality of the model's embeddings.
        
    Returns:
        A numpy array of shape (position, d_model) in float16, 
        or -1 if constraints are not met.
    """
    
    # 1. Edge cases handling as per requirement
    if position == 0 or d_model <= 0:
        return -1
    
    # 2. Initialize the Positional Encoding matrix
    # Shape: (seq_len, d_model)
    pe = np.zeros((position, d_model))
    
    # 3. Create a vector of positions
    # Shape: (position, 1)
    pos_vec = np.arange(position)[:, np.newaxis]
    
    # 4. Create a vector of dimension indices (only even indices are needed for the denominator)
    # i ranges from 0 to d_model, incrementing by 2.
    # Shape: (1, d_model // 2)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # 5. Calculate Sin and Cos components
    # Even indices (0, 2, 4, ...): use Sine
    pe[:, 0::2] = np.sin(pos_vec * div_term)
    
    # Odd indices (1, 3, 5, ...): use Cosine
    # Note: If d_model is odd, the sine and cosine slices must match.
    # Standard transformers use even d_model.
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(pos_vec * div_term)
    else:
        # If d_model is odd, the last cosine slice will be one shorter
        pe[:, 1::2] = np.cos(pos_vec * div_term[:d_model//2])
    
    # 6. Convert to float16 and return
    pe = pe.astype(np.float16)
    return pe

if __name__ == "__main__":
    # Example Usage and Verification
    # -----------------------------------------------------------
    seq_len = 10
    d_model = 16
    
    print(f"Generating Positional Encoding for seq_len={seq_len}, d_model={d_model}...")
    encoding = pos_encoding(seq_len, d_model)
    
    if isinstance(encoding, np.ndarray):
        print("\nPositional Encoding Matrix (first 2 rows):")
        print(encoding[:2])
        print(f"\nMatrix Shape: {encoding.shape}")
        print(f"Data Type: {encoding.dtype}")
        
        # Verify even/odd indices property
        # PE(pos, 0) should be sin(pos/1), PE(pos, 1) should be cos(pos/1)
        # For pos=1: PE(1, 0) ~ sin(1) ~ 0.841, PE(1, 1) ~ cos(1) ~ 0.540
        print(f"\nVerification at pos=1:")
        print(f"pos_encoding(1, 0): {encoding[1, 0]:.4f} (Expected sin(1) ≈ 0.8415)")
        print(f"pos_encoding(1, 1): {encoding[1, 1]:.4f} (Expected cos(1) ≈ 0.5403)")
    else:
        print(f"Error: {encoding}")
    
    # Test Edge Cases
    print("\nTesting Edge Cases:")
    print(f"pos_encoding(0, 16): {pos_encoding(0, 16)}")
    print(f"pos_encoding(10, 0): {pos_encoding(10, 0)}")