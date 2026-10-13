import torch
import torch.nn as nn

class FeatureDeconfounder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sigma_inv = None
    
    def fit(self, metadata):
        X = metadata
        reg = 1e-5
        N, K = X.shape
        I = torch.eye(K, device=X.device, dtype=X.dtype)
        cov = X.t() @ X + reg * I
        self.Sigma_inv = torch.linalg.inv(cov)
    
    def transform(self, features, metadata):
        X = metadata
        f = features
        beta_hat = self.Sigma_inv @ (X.t() @ f)
        f_res = f - X @ beta_hat
        return f_res
