import torch

def flow_epe(pred, gt, mask=None, max_flow=None):
    pred = torch.tensor(pred, dtype=float)
    gt = torch.tensor(gt, dtype=float)
    
    epe = torch.sqrt(torch.sum((pred - gt)**2, dim=-1))
    valid = torch.ones_like(epe, dtype=bool)
    
    if mask is not None:
        valid &= torch.tensor(mask).astype(bool)
        
    if max_flow is not None:
        v_flow = torch.sqrt(torch.sum(gt**2, dim=-1))
        valid &= (v_flow <= max_flow)
        
    if not np.any(valid):
        return 0.0
        
    return float(torch.mean(epe[valid]))
