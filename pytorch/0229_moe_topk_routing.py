"""Deep-ML #229: Sparse MoE Top-K Routing.
https://www.deep-ml.com/problems/229
Standalone PyTorch solution; public-example replay is not an official submission.
"""
import torch

def moe_topk_routing(router_logits, expert_outputs, k):
    if not 1 <= k <= router_logits.shape[-1]:
        raise ValueError('k must be between 1 and num_experts.')
    values, indices = torch.topk(router_logits, k, dim=-1)
    weights = torch.softmax(values, dim=-1)
    index = indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
    selected = torch.gather(expert_outputs, dim=1, index=index)
    return (selected * weights.unsqueeze(-1)).sum(dim=1)
