def float_precision_comparison(num: list, format_type: str) -> list:
    import numpy as np
    num = np.array(num, dtype=float)
    if format_type.lower() == 'fp16':
        return np.round(num.astype(np.float16).astype(float), 4).tolist()
    elif format_type.lower() == 'bf16':
        # BFloat16 roughly truncates the 16 LSBs of FP32 mantissa. 
        # NumPy doesn't have native bfloat16, simulate by cutting precision.
        int32_repr = num.astype(np.float32).view(np.uint32)
        int32_repr &= 0xFFFF0000
        return np.round(int32_repr.view(np.float32), 4).tolist()
    elif format_type.lower() in ('fp8', 'fp4'):
        # Mocking extreme quantizations generically by massive rounding scale
        scale = 100.0 if format_type.lower() == 'fp8' else 10.0
        return np.round(np.round(num * scale) / scale, 4).tolist()
    return np.round(num, 4).tolist()
