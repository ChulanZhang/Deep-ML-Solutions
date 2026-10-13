import os

files_content = {
    "44-Leaky ReLU Activation Function.py": '''import torch
import torch.nn.functional as F

def leaky_relu(x: list[float], negative_slope: float = 0.01) -> list[float]:
    t = torch.tensor(x, dtype=torch.float32)
    return F.leaky_relu(t, negative_slope=negative_slope).tolist()
''',

    "46-Implement Precision Metric.py": '''import torch

def precision_score(y_true: list[int], y_pred: list[int]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.float32)
    t_pred = torch.tensor(y_pred, dtype=torch.float32)
    tp = torch.sum((t_true == 1) & (t_pred == 1)).float()
    fp = torch.sum((t_true == 0) & (t_pred == 1)).float()
    if (tp + fp) == 0: return 0.0
    return (tp / (tp + fp)).item()
''',

    "52-Implement Recall Metric in Binary Classification.py": '''import torch

def recall_score(y_true: list[int], y_pred: list[int]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.float32)
    t_pred = torch.tensor(y_pred, dtype=torch.float32)
    tp = torch.sum((t_true == 1) & (t_pred == 1)).float()
    fn = torch.sum((t_true == 1) & (t_pred == 0)).float()
    if (tp + fn) == 0: return 0.0
    return (tp / (tp + fn)).item()
''',

    "53-Implement Self-Attention Mechanism.py": '''import torch
import torch.nn as nn
import math

def self_attention(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> list[list[float]]:
    # PyTorch 原生手动计算
    Q = torch.tensor(q, dtype=torch.float32)
    K = torch.tensor(k, dtype=torch.float32)
    V = torch.tensor(v, dtype=torch.float32)
    d_k = Q.shape[-1]
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(weights, V).tolist()
''',

    "56-KL Divergence Between Two Normal Distributions.py": '''import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

def kl_divergence_normal(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """
    使用 PyTorch 自带的 KL Divergence 接口
    """
    p = Normal(torch.tensor(mu1, dtype=torch.float32), torch.tensor(sigma1, dtype=torch.float32))
    q = Normal(torch.tensor(mu2, dtype=torch.float32), torch.tensor(sigma2, dtype=torch.float32))
    return kl_divergence(p, q).item()
''',

    "61-Implement F-Score Calculation for Binary Classification.py": '''import torch

def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.float32)
    t_pred = torch.tensor(y_pred, dtype=torch.float32)
    tp = torch.sum((t_true == 1) & (t_pred == 1)).float()
    fp = torch.sum((t_true == 0) & (t_pred == 1)).float()
    fn = torch.sum((t_true == 1) & (t_pred == 0)).float()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    
    if precision + recall == 0: return 0.0
    return (2 * precision * recall / (precision + recall)).item()
''',

    "63-Binary Classification with Logistic Regression.py": '''import torch

def logistic_regression(X: list[list[float]], y: list[float], lr: float, epochs: int) -> tuple:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    w = torch.zeros((X_t.shape[1], 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    optimizer = torch.optim.SGD([w, b], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = X_t @ w + b
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        
    return w.detach().flatten().tolist(), b.item()
''',

    "64-Pegasos Kernel SVM Implementation.py": '''import torch

def pegasos_svm(X: list[list[float]], y: list[int], lambda_val: float, epochs: int) -> list[float]:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    w = torch.zeros(X_t.shape[1], dtype=torch.float32)
    
    for _ in range(epochs):
        for i in range(len(X_t)):
            lr = 1.0 / (lambda_val * (_ * len(X_t) + i + 1))
            margin = y_t[i] * torch.dot(w, X_t[i])
            if margin < 1:
                w = (1 - lr * lambda_val) * w + lr * y_t[i] * X_t[i]
            else:
                w = (1 - lr * lambda_val) * w
    return w.tolist()
''',

    "66-Calculate Jaccard Index for Binary Classification.py": '''import torch

def jaccard_index(y_true: list[int], y_pred: list[int]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.bool)
    t_pred = torch.tensor(y_pred, dtype=torch.bool)
    
    intersection = torch.logical_and(t_true, t_pred).sum().float()
    union = torch.logical_or(t_true, t_pred).sum().float()
    if union == 0: return 0.0
    return (intersection / union).item()
''',

    "69-Calculate R-squared for Regression Analysis.py": '''import torch

def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.float32)
    t_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    ss_res = torch.sum((t_true - t_pred) ** 2)
    ss_tot = torch.sum((t_true - t_true.mean()) ** 2)
    
    return (1 - ss_res / ss_tot).item()
''',

    "71-Calculate Root Mean Square Error (RMSE).py": '''import torch
import torch.nn.functional as F

def rmse(y_true: list[float], y_pred: list[float]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.float32)
    t_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    mse = F.mse_loss(t_pred, t_true)
    return torch.sqrt(mse).item()
''',

    "76-Calculate Cosine Similarity Between Vectors.py": '''import torch
import torch.nn.functional as F

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    t1 = torch.tensor(v1, dtype=torch.float32)
    t2 = torch.tensor(v2, dtype=torch.float32)
    
    return F.cosine_similarity(t1, t2, dim=0).item()
''',

    "83-Dot Product Calculator Easy.py": '''import torch

def dot_product(v1: list[float], v2: list[float]) -> float:
    t1 = torch.tensor(v1, dtype=torch.float32)
    t2 = torch.tensor(v2, dtype=torch.float32)
    return torch.dot(t1, t2).item()
''',

    "84-Convert Vector to Diagonal Matrix.py": '''import torch

def vector_to_diagonal(v: list[float]) -> list[list[float]]:
    t = torch.tensor(v, dtype=torch.float32)
    return torch.diag(t).tolist()
''',

    "85-Find the column space of a matrix.py": '''import torch

def column_space(matrix: list[list[float]]) -> list[list[float]]:
    mat_t = torch.tensor(matrix, dtype=torch.float32)
    # Perform SVD to find the column space basis (left singular vectors)
    U, S, Vh = torch.linalg.svd(mat_t)
    rank = torch.linalg.matrix_rank(mat_t)
    return U[:, :rank].tolist()
''',

    "85-Positional Encoding Calculator.py": '''import torch
import math

def positional_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
        
    return pe.tolist()
'''
}

out_dir = "pytorch"
import os
os.makedirs(out_dir, exist_ok=True)

for name, content in files_content.items():
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(content)

print("Batch 4 & 5 PyTorch completed!")
