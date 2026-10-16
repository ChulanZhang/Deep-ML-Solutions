def count_parameters(layers: list) -> int:
    total = 0
    for layer in layers:
        if isinstance(layer, dict):
            fan_in = layer.get('in', 0)
            fan_out = layer.get('out', 0)
            bias = layer.get('bias', True)
            total += fan_in * fan_out + (fan_out if bias else 0)
        elif isinstance(layer, list) or isinstance(layer, tuple):
            if len(layer) == 2:
                total += layer[0] * layer[1] + layer[1]
    return total
