import numpy as np

def neural_memory_update(memory: np.ndarray, new_info: np.ndarray, surprise_score: float, momentum: float) -> np.ndarray:
    memory = np.array(memory, dtype=float)
    new_info = np.array(new_info, dtype=float)
    
    updated_memory = momentum * memory + surprise_score * new_info
    return np.round(updated_memory, 4)
