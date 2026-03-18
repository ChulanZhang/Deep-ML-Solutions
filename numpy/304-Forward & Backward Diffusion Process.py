import numpy as np

def diffusion_processes(x0, t, alphas, alphas_cumprod, noise, pred_noise, z):
    x0 = np.array(x0, dtype=float)
    noise = np.array(noise, dtype=float)
    pred_noise = np.array(pred_noise, dtype=float)
    z = np.array(z, dtype=float)
    
    # Forward step
    alpha_bar = alphas_cumprod[t]
    xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
    
    # Backward step
    alpha = alphas[t]
    
    if t > 0:
        sigma_t = np.sqrt((1 - alphas_cumprod[t-1]) / (1 - alpha_bar) * (1 - alpha))
    else:
        sigma_t = 0.0
        
    x_prev = (1.0 / np.sqrt(alpha)) * (xt - ((1 - alpha) / np.sqrt(1 - alpha_bar)) * pred_noise)
    if t > 0:
        x_prev += sigma_t * z
        
    return np.round(xt, 4), np.round(x_prev, 4)
