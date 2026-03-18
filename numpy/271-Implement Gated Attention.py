import numpy as np

def gated_attention(x: np.ndarray, w_gate: np.ndarray, w_attn: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    w_gate = np.array(w_gate, dtype=float)
    w_attn = np.array(w_attn, dtype=float)
    
    gate_logits = np.dot(x, w_gate)
    gate = 1.0 / (1.0 + np.exp(-gate_logits))
    
    attn_out = np.dot(x, w_attn)
    return np.round(gate * attn_out, 4)
