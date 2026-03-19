def eagle_draft_model(hidden_states: list, W_draft: list) -> list:
    import numpy as np
    H = np.array(hidden_states, dtype=float)
    W = np.array(W_draft, dtype=float)
    
    # Simple linear projector mapping hidden states to a drafted token logits
    logits = np.dot(H, W)
    best_tokens = np.argmax(logits, axis=-1)
    
    return best_tokens.tolist()
