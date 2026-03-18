import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    """
    Implement the forward pass of a DenseNet dense block.
    Input shape is (N, H, W, C).
    Kernels shape assumed: (num_layers, kernel_size[0], kernel_size[1], current_c, growth_rate)
    """
    x = np.array(input_data, dtype=float)
    kh, kw = kernel_size
    pad_h = kh // 2
    pad_w = kw // 2
    
    current_x = x
    
    for i in range(num_layers):
        N, H, W, C = current_x.shape
        
        # 1. ReLU
        activated = np.maximum(0, current_x)
        
        # 2. 2D Conv
        w = np.array(kernels[i], dtype=float)
        # Assuming w shape is (kh, kw, C, growth_rate)
        # Pad activated
        padded = np.pad(activated, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant')
        
        conv_out = np.zeros((N, H, W, growth_rate), dtype=float)
        
        for h_out in range(H):
            for w_out in range(W):
                # region shape: (N, kh, kw, C)
                region = padded[:, h_out:h_out+kh, w_out:w_out+kw, :]
                
                # We want dot product of region with w. 
                # region is (N, kh, kw, C), w is (kh, kw, C, growth_rate)
                # multiply element-wise over kh, kw, C and sum
                # or computationally:
                # region reshaped to (N, kh*kw*C)
                # w reshaped to (kh*kw*C, growth_rate)
                region_flat = region.reshape(N, -1)
                w_flat = w.reshape(-1, growth_rate)
                
                conv_out[:, h_out, w_out, :] = region_flat @ w_flat
                
        # 3. Concatenate
        current_x = np.concatenate((current_x, conv_out), axis=-1)
        
    return current_x
