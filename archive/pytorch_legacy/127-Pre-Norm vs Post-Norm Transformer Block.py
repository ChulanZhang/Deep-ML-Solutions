import torch

def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta

def transformer_block(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, 
                      gamma1: torch.Tensor, beta1: torch.Tensor, gamma2: torch.Tensor, beta2: torch.Tensor, 
                      mode: str, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply a transformer block with two sublayers using either pre-norm or post-norm.
    Assumes Sublayer 1 is a simple linear projection (simulating Attention without QKV complexity for this block) -> Actually just applying W1, b1.
    Wait, let's assume Sublayer1 is: z = x @ W1 + b1
    Sublayer2 is: z = x @ W2 + b2
    """
    if mode == 'pre':
        # Pre-Norm
        # Sublayer 1
        x_norm1 = layer_norm(x, gamma1, beta1, eps)
        sub1_out = torch.matmul(x_norm1, W1) + b1
        x = x + sub1_out
        
        # Sublayer 2
        x_norm2 = layer_norm(x, gamma2, beta2, eps)
        sub2_out = torch.matmul(x_norm2, W2) + b2
        x = x + sub2_out
        
    elif mode == 'post':
        # Post-Norm
        # Sublayer 1
        sub1_out = torch.matmul(x, W1) + b1
        x = layer_norm(x + sub1_out, gamma1, beta1, eps)
        
        # Sublayer 2
        sub2_out = torch.matmul(x, W2) + b2
        x = layer_norm(x + sub2_out, gamma2, beta2, eps)
        
    return x
