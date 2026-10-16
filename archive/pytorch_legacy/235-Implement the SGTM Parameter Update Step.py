import torch

def sgtm_update(param, grad, momentum_buffer, lr=0.01, momentum=0.9):
    # Stand-in structural SGD momentum
    v = momentum * torch.tensor(momentum_buffer) + torch.tensor(grad)
    p = torch.tensor(param) - lr * v
    if np.isscalar(param) or (isinstance(param, torch.Tensor) and param.ndim == 0):
        return float(p), float(v)
    return p, v
