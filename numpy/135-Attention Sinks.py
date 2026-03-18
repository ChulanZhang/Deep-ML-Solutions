import numpy as np

def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
    """
    Identify token positions that act as attention sinks.
    attn_weights: (num_heads, seq_len, seq_len) -> softmax applied over last dim
    """
    num_heads, seq_len, _ = attn_weights.shape
    
    # Average attention mass received by each token (average over all queries and heads)
    # Target dim: sum over queries (dim=1) and heads (dim=0), then divide by (heads * queries)
    # Actually, the problem asks for avg attention received per position
    
    avg_attention = np.mean(attn_weights, axis=(0, 1))
    
    sink_positions = np.where(avg_attention >= threshold)[0].tolist()
    
    avg_attn_list = [round(float(v), 4) for v in avg_attention]
    sink_scores = [avg_attn_list[i] for i in sink_positions]
    
    return {
        'sink_positions': sink_positions,
        'avg_attention_received': avg_attn_list,
        'sink_scores': sink_scores
    }
