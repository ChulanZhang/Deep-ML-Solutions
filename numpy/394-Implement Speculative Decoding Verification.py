import numpy as np

def speculative_decoding_verify(target_probs: np.ndarray, draft_probs: np.ndarray, draft_tokens: list, rands: list) -> list:
    accepted = []
    for i in range(len(draft_tokens)):
        token = draft_tokens[i]
        p_t = target_probs[i, token]
        p_d = draft_probs[i, token]
        
        ratio = p_t / (p_d + 1e-10)
        if rands[i] < ratio:
            accepted.append(token)
        else:
            break
            
    return accepted
