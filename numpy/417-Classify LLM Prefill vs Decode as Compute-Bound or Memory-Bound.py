def classify_llm_phase(batch_size: int, seq_len: int, model_params: int, hardware_flops: float, hardware_bw: float) -> str:
    # Model param bytes (assume FP16)
    bytes_accessed = model_params * 2
    
    # FLOPs approximation for forward pass over sequence
    flops_needed = 2 * model_params * batch_size * seq_len
    
    arithmetic_intensity = flops_needed / bytes_accessed
    hardware_intensity = hardware_flops / hardware_bw
    
    if arithmetic_intensity < hardware_intensity:
        return "Memory-Bound"
    else:
        return "Compute-Bound"
