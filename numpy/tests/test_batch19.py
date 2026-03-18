import sys
import numpy as np
import importlib.util
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
numpy_dir = os.path.dirname(script_dir)

def load_module(name, filename):
    path = os.path.join(numpy_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod384 = load_module("m384", "384-Contrastive Loss (InfoNCE _ SimCLR-style).py")
mod386 = load_module("m386", "386-Spectral Normalization.py")
mod387 = load_module("m387", "387-Triplet Margin Loss.py")
mod388 = load_module("m388", "388-Sliding Window Attention.py")
mod390 = load_module("m390", "390-Implement Multiquery Attention (MQA).py")
mod391 = load_module("m391", "391-Implement Grouped Query Attention (GQA).py")
mod393 = load_module("m393", "393-Implement Variational Autoencoder (VAE) Loss (ELBO).py")
mod394 = load_module("m394", "394-Implement Speculative Decoding Verification.py")
mod395 = load_module("m395", "395-DDPM Noise Schedule (Linear Beta Schedule).py")
mod396 = load_module("m396", "396-Implement DDPM Reverse Sampling Step.py")
mod397 = load_module("m397", "397-Classifier-Free Guidance for Conditional Diffusion.py")
mod398 = load_module("m398", "398-DDIM Deterministic Sampling Step.py")
mod399 = load_module("m399", "399-Diffusion Model U-Net Time Embedding.py")
mod400 = load_module("m400", "400-Noise Prediction Loss for Diffusion Training.py")
mod401 = load_module("m401", "401-Exponential Moving Average (EMA) for Diffusion Model Weights.py")

def check():
    assert isinstance(mod384.infonce_loss([[1, 0]], [[1, 0]]), float)
    
    sn_W, sn_u = mod386.spectral_normalization(np.random.randn(4, 4), np.random.randn(4))
    assert sn_W.shape == (4, 4)
    
    assert isinstance(mod387.triplet_margin_loss([1], [1], [-1]), float)
    assert hasattr(mod388, 'sliding_window_attention')
    
    Q = np.random.randn(2, 4, 10, 8)
    K = np.random.randn(2, 1, 10, 8)
    V = np.random.randn(2, 1, 10, 8)
    assert mod390.multiquery_attention(Q, K, V).shape == (2, 4, 10, 8)
    
    assert hasattr(mod391, 'grouped_query_attention')
    
    assert isinstance(mod393.vae_loss(1.0, [0], [0]), float)
    
    acc = mod394.speculative_decoding_verify(np.array([[0.9, 0.1]]), np.array([[0.8, 0.2]]), [0], [0.5])
    assert isinstance(acc, list)
    
    b, a, ac = mod395.linear_beta_schedule(10)
    assert len(b) == 10
    
    xt = mod396.ddpm_reverse_step([1.0], [0.1], 1, a, ac, b, [0.1])
    assert len(xt) == 1
    
    cfg = mod397.classifier_free_guidance([0.1], [0.5], 2.0)
    assert len(cfg) == 1
    
    ddim = mod398.ddim_step([1.0], [0.1], 1, ac)
    assert len(ddim) == 1
    
    emb = mod399.unet_time_embedding([1, 2], 16)
    assert emb.shape == (2, 16)
    
    assert isinstance(mod400.noise_prediction_loss([0], [0]), float)
    
    ema = mod401.ema_weights([1.0], [0.5], 0.9)
    assert len(ema) == 1
    
    print("Batch 19 OK")

if __name__ == '__main__':
    check()
