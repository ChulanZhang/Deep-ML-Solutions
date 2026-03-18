import numpy as np

def gambler_value_iteration(ph, theta=1e-9):
    V = np.zeros(101)
    V[100] = 1.0  # Goal state
    policy = np.zeros(101, dtype=int)
    
    while True:
        delta = 0.0
        for s in range(1, 100):
            v_old = V[s]
            
            # Actions: bets from 1 to min(s, 100-s)
            stakes = np.arange(1, min(s, 100 - s) + 1)
            # Bellman backup: Value = ph * V[s + a] + (1 - ph) * V[s - a]
            
            # Vectorized evaluation for all stakes
            v_wins = V[s + stakes]
            v_loses = V[s - stakes]
            returns = ph * v_wins + (1 - ph) * v_loses
            
            best_val = np.max(returns)
            V[s] = best_val
            
            delta = max(delta, abs(v_old - best_val))
            
        if delta < theta:
            break
            
    # Compute deterministic policy resolving ties stably
    for s in range(1, 100):
        stakes = np.arange(1, min(s, 100 - s) + 1)
        v_wins = V[s + stakes]
        v_loses = V[s - stakes]
        returns = ph * v_wins + (1 - ph) * v_loses
        
        # Round slightly to tackle float drifting equality
        returns = np.round(returns, 5)
        best_action_idx = np.argmax(returns)
        policy[s] = stakes[best_action_idx]
        
    return V, policy
