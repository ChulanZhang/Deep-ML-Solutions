import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples.
        k: Number of folds.
        shuffle: Whether to shuffle indices before splitting.
        
    Returns:
        A list of k tuples, each containing (train_indices, test_indices).
    """
    # 1. Create array of indices
    indices = np.arange(n_samples)
    
    # 2. Shuffle if requested
    if shuffle:
        np.random.shuffle(indices)
        
    # 3. Calculate fold sizes
    # Some folds might be larger by 1 if n_samples is not perfectly divisible by k
    base_size = n_samples // k
    remainder = n_samples % k
    
    folds = []
    current = 0
    for i in range(k):
        # Add 1 to fold size if there are remaining elements to distribute
        fold_size = base_size + 1 if i < remainder else base_size
        
        # Test indices are the current fold
        test_indices = indices[current : current + fold_size].tolist()
        
        # Train indices are everything else
        train_indices = np.concatenate((indices[:current], indices[current + fold_size:])).tolist()
        
        folds.append((train_indices, test_indices))
        
        current += fold_size
        
    return folds
