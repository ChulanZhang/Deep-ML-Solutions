import numpy as np

def ucb_action(counts, values, t, c):
    """
    Choose an action using the UCB1 formula.
    """
    counts = np.array(counts, dtype=float)
    values = np.array(values, dtype=float)
    
    # Handle the unvisited actions (count == 0)
    if np.any(counts == 0):
        return int(np.where(counts == 0)[0][0])
        
    ucb_values = values + c * np.sqrt(np.log(t) / counts)
    return int(np.argmax(ucb_values))
