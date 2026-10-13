import torch

def topk_routing(x: torch.Tensor, w_gate: torch.Tensor, k: int) -> tuple:
    x_arr = torch.tensor(x, dtype=float)
    w_gate_arr = torch.tensor(w_gate, dtype=float)
    
    # Calculate initial scores
    scores = torch.matmul(x_arr, w_gate_arr)
    
    # Sort backwards to get top-k
    sorted_idx = np.argsort(scores, dim=-1)
    top_indices = sorted_idx[:, -k:]    # The last k elements are the largest
    top_indices = top_indices[:, ::-1]  # Reverse them so biggest is first
    
    # Extract scores
    top_scores = np.take_along_axis(scores, top_indices, dim=1)
    
    # Compute Softmax strictly over the top_scores
    max_scores = torch.amax(top_scores, dim=1, keepdim=True)
    exp_scores = torch.exp(top_scores - max_scores)
    
    gating_weights = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
    
    return torch.round(gating_weights, 4), top_indices
