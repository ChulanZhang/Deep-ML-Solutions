def count_parameters(layers: list) -> int:
    """
    Calculate Number of Parameters in Neural Network
    """
    total = 0
    for layer in layers:
        l_type = layer.get('type')
        if l_type == 'dense':
            in_f = layer['input_dim']
            out_f = layer['output_dim']
            total += in_f * out_f + (out_f if layer.get('bias', True) else 0)
        elif l_type == 'conv2d':
            in_f = layer['input_channels']
            out_f = layer['output_channels']
            kh, kw = layer['kernel_size']
            total += kh * kw * in_f * out_f + (out_f if layer.get('bias', True) else 0)
        elif l_type == 'embedding':
            num = layer['vocab_size']
            dim = layer['embedding_dim']
            total += num * dim
            
    return total
