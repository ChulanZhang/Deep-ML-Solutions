import torch

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: torch.Tensor of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    if torch.rand() < epsilon:
        return int(np.random.randint(len(Q)))
    return int(np.argmax(Q))
