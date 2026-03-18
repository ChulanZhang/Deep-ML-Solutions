import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    """
    Implement a simple 2D convolutional layer.
    
    Args:
        input_matrix: 2D numpy array representing the input image
        kernel: 2D numpy array representing the filter
        padding: integer representing the padding size
        stride: integer representing the stride step
        
    Returns:
        output_matrix: 2D numpy array resulting from the convolution
    """
    # 1. Apply zero padding to the input matrix if padding > 0
    if padding > 0:
        input_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    
    # 2. Calculate output dimensions based on the convolution formula:
    # Output_size = (Input_size - Kernel_size) // Stride + 1
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    
    output_matrix = np.zeros((output_height, output_width))
    
    # 3. Slide the kernel over the padded input matrix
    for i in range(output_height):
        for j in range(output_width):
            # Calculate the starting and ending indices for the current window
            row_start = i * stride
            row_end = row_start + kernel_height
            col_start = j * stride
            col_end = col_start + kernel_width
            
            # Extract the region of interest
            region = input_matrix[row_start:row_end, col_start:col_end]
            
            # Perform element-wise multiplication and sum (the actual convolution/cross-correlation operation)
            output_matrix[i, j] = np.sum(region * kernel)
            
    return output_matrix
