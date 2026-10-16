"""Deep-ML #409: MoE with Shared Expert Forward Pass.
https://www.deep-ml.com/problems/409
Standalone PyTorch solution; public-example replay is not an official submission.
Tie policy follows the PyTorch public examples; the shared NumPy description says the opposite.
"""
import torch

def moe_shared_expert_forward(X, W_gate, W_shared, W_experts, top_k=2):
    logits = X @ W_gate
    if not 1 <= top_k <= len(W_experts):
        raise ValueError('top_k is outside the expert range.')
    # PyTorch public examples choose the lower index when scores tie.
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    values = logits.gather(-1, indices)
    weights = torch.softmax(values, dim=-1)
    selected = torch.stack(W_experts)[indices]
    outputs = torch.einsum('nd,nkdf->nkf', X, selected)
    routed = (outputs * weights.unsqueeze(-1)).sum(dim=1)
    shared = X @ W_shared
    return {'output': shared + routed, 'shared_output': shared,
            'routed_output': routed, 'routing_indices': indices,
            'routing_weights': weights}
