"""Deep-ML #384: Contrastive Loss (InfoNCE / SimCLR-style).
https://www.deep-ml.com/problems/384
Standalone PyTorch solution; public-example replay is not an official submission.
Adjacent rows are positives. The official return type is float, not a training loss Tensor.
"""
import torch
import torch.nn.functional as F

def contrastive_loss(embeddings, temperature):
    count = embeddings.shape[0]
    if temperature <= 0 or count < 2 or count % 2:
        raise ValueError('Need positive temperature and an even positive pair count.')
    z = F.normalize(embeddings, dim=-1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(count, device=z.device, dtype=torch.bool), float('-inf'))
    labels = torch.arange(count, device=z.device) ^ 1
    return F.cross_entropy(logits, labels).item()
