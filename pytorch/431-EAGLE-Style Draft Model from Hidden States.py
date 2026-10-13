def eagle_draft_model(hidden_states: list, W_draft: list) -> list:
    import torch
    H = torch.tensor(hidden_states, dtype=float)
    W = torch.tensor(W_draft, dtype=float)
    
    # Simple linear projector mapping hidden states to a drafted token logits
    logits = torch.matmul(H, W)
    best_tokens = np.argmax(logits, dim=-1)
    
    return best_tokens.tolist()
