import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    V: state value vector (list or array)
    transitions: mapping of state -> mapping of action -> list of (prob, next_s, reward) or equivalent.
    """
    V_arr = np.array(V, dtype=float)
    V_new = np.zeros_like(V_arr)
    
    for s in range(len(V_arr)):
        max_val = float('-inf')
        
        # Check type of transitions structured
        actions = transitions[s] if isinstance(transitions, list) else transitions.get(s, {})
        if not actions:
            max_val = V_arr[s]
        else:
            action_iter = actions.values() if isinstance(actions, dict) else actions
            
            for action_transitions in action_iter:
                q_val = 0.0
                for prob, next_s, reward in action_transitions:
                    q_val += prob * (reward + gamma * V_arr[next_s])
                
                if q_val > max_val:
                    max_val = q_val
                    
        V_new[s] = max_val if max_val != float('-inf') else V_arr[s]
        
    if isinstance(V, list):
        return V_new.tolist()
    return V_new
