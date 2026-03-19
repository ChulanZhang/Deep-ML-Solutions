import numpy as np

def shared_expert_moe(x: np.ndarray, shared_W: np.ndarray, expert_Ws: list, gate_W: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    shared_W = np.array(shared_W, dtype=float)
    gate_W = np.array(gate_W, dtype=float)
    
    shared_out = np.dot(x, shared_W)
    
    logits = np.dot(x, gate_W)
    max_l = np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(logits - max_l)
    gates = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    
    expert_out = np.zeros_like(shared_out)
    for i, W in enumerate(expert_Ws):
        W = np.array(W, dtype=float)
        expert_out += gates[:, i:i+1] * np.dot(x, W)
        
    return np.round(shared_out + expert_out, 4)
