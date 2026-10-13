def roofline_analysis(peak_gflops: float, peak_bandwidth_gbs: float, operations: list) -> dict:
    """
    Perform Roofline Model analysis for GPU operations.
    """
    ridge_point = peak_gflops / peak_bandwidth_gbs if peak_bandwidth_gbs > 0 else float('inf')
    
    results = []
    for op in operations:
        intensity = op['flops'] / op['bytes'] if op['bytes'] > 0 else float('inf')
        attainable = min(peak_gflops, intensity * peak_bandwidth_gbs)
        bottleneck = 'compute-bound' if intensity >= ridge_point else 'memory-bound'
        efficiency = (attainable / peak_gflops) * 100
        
        results.append({
            'name': op['name'],
            'operational_intensity': intensity,
            'attainable_gflops': attainable,
            'bottleneck': bottleneck,
            'efficiency': efficiency
        })
        
    return {
        'ridge_point': ridge_point,
        'operations': results
    }
