import numpy as np

def descriptive_statistics(data: list) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.

    Args:
        data: List or numpy array of numerical values

    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation, 
        percentiles (25th, 50th, 75th), and interquartile range (IQR).
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    n = len(data)
    if n == 0:
        return {}
        
    # Mean
    mean_val = np.sum(data) / n
    
    # Median
    sorted_data = np.sort(data)
    mid = n // 2
    if n % 2 == 0:
        median_val = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    else:
        median_val = sorted_data[mid]
        
    # Mode (using unique counts)
    vals, counts = np.unique(data, return_counts=True)
    mode_val = vals[np.argmax(counts)]
    
    # Population Variance & Std Dev (divide by N)
    var_val = np.sum((data - mean_val) ** 2) / n
    std_val = np.sqrt(var_val)
    
    # Percentiles
    # For coding interviews, using np.percentile is often acceptable/preferred if numpy is imported.
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50) # Same as median
    p75 = np.percentile(data, 75)
    iqr = p75 - p25

    return {
        'mean': float(mean_val),
        'median': float(median_val),
        'mode': float(mode_val),
        'variance': float(var_val),
        'standard_deviation': float(std_val),
        '25th_percentile': float(p25),
        '50th_percentile': float(p50),
        '75th_percentile': float(p75),
        'interquartile_range': float(iqr)
    }
