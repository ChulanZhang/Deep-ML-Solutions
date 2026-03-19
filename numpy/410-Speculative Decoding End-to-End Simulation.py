import numpy as np

def speculative_decoding_sim(steps: int, draft_accept_rate: float, draft_time: float, target_time: float) -> tuple:
    # Simulates the latency of speculative decoding vs standard
    # Under simplification, each step attempts to draft K tokens
    # Average accepted tokens per step is 1 + draft_accept_rate * K
    # For a normalized K=3 length:
    K = 3
    tokens_per_draft = 1 + draft_accept_rate * K
    
    # Expected time per speculative step = K * draft_time + target_time
    time_per_spec_step = K * draft_time + target_time
    
    # Number of speculative steps needed
    spec_steps = np.ceil(steps / tokens_per_draft)
    total_spec_time = spec_steps * time_per_spec_step
    
    # Standard time
    total_standard_time = steps * target_time
    
    return np.round(total_spec_time, 4), np.round(total_standard_time, 4)
