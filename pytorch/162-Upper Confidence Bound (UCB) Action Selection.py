import torch

def ucb_action(counts, values, t, c):
    """
    Choose an action using the UCB1 formula.
    """
    counts = torch.tensor(counts, dtype=float)
    values = torch.tensor(values, dtype=float)
    
    # Handle the unvisited actions (count == 0)
    if np.any(counts == 0):
        return int(np.where(counts == 0)[0][0])
        
    ucb_values = values + c * torch.sqrt(torch.log(t) / counts)
    return int(np.argmax(ucb_values))
