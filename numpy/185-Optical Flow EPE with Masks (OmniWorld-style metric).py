import numpy as np

def flow_epe(pred, gt, mask=None, max_flow=None):
    pred = np.array(pred, dtype=float)
    gt = np.array(gt, dtype=float)
    
    epe = np.sqrt(np.sum((pred - gt)**2, axis=-1))
    valid = np.ones_like(epe, dtype=bool)
    
    if mask is not None:
        valid &= np.array(mask).astype(bool)
        
    if max_flow is not None:
        v_flow = np.sqrt(np.sum(gt**2, axis=-1))
        valid &= (v_flow <= max_flow)
        
    if not np.any(valid):
        return 0.0
        
    return float(np.mean(epe[valid]))
