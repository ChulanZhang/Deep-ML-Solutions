import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    P: Transition dict
    R: Reward dict (assumed identical structure)
    V: state value vector
    gamma: discount factor
    """
    expected = 0.0
    transitions = P.get(state, {}).get(action, {})
    
    for next_s, prob in transitions.items():
        reward = R.get(state, {}).get(action, {}).get(next_s, 0.0) if R else 0.0
        expected += prob * (reward + gamma * V[next_s])
        
    return float(expected)
