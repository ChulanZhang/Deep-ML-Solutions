import os

files_content = {
    "86-Reshape Matrix.py": '''import torch

def reshape_matrix(a: list[list[float]], new_shape: tuple[int, int]) -> list[list[float]]:
    t = torch.tensor(a, dtype=torch.float32)
    return t.reshape(new_shape).tolist()
''',

    "87-Implement Compressed Row Sparse Matrix (CSR) Format Conversion.py": '''import torch

def dense_to_csr(matrix: list[list[float]]) -> dict:
    t = torch.tensor(matrix, dtype=torch.float32)
    csr = t.to_sparse_csr()
    return {
        "crow_indices": csr.crow_indices().tolist(),
        "col_indices": csr.col_indices().tolist(),
        "values": csr.values().tolist()
    }
''',

    "88-Implement Orthogonal Projection of a Vector onto a Line.py": '''import torch

def orthogonal_projection(v: list[float], L: list[float]) -> list[float]:
    v_t = torch.tensor(v, dtype=torch.float32)
    L_t = torch.tensor(L, dtype=torch.float32)
    
    proj = (torch.dot(v_t, L_t) / torch.dot(L_t, L_t)) * L_t
    return proj.tolist()
''',

    "89-Implement Compressed Column Sparse Matrix Format.py": '''import torch

def dense_to_csc(matrix: list[list[float]]) -> dict:
    t = torch.tensor(matrix, dtype=torch.float32)
    csc = t.to_sparse_csc()
    return {
        "ccol_indices": csc.ccol_indices().tolist(),
        "row_indices": csc.row_indices().tolist(),
        "values": csc.values().tolist()
    }
''',

    "90-Transformation Matrix from Basis B to C.py": '''import torch

def basis_transformation(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    B_t = torch.tensor(B, dtype=torch.float32)
    C_t = torch.tensor(C, dtype=torch.float32)
    # Solve C * T = B => T = C^-1 * B
    # Note: T is defined such that coordinates change as [v]_C = T [v]_B
    T = torch.linalg.solve(C_t, B_t)
    return T.tolist()
''',

    "91-Matrix Transformation.py": '''import torch

def transform_matrix(T: list[list[float]], S: list[list[float]]) -> list[list[float]]:
    T_t = torch.tensor(T, dtype=torch.float32)
    S_t = torch.tensor(S, dtype=torch.float32)
    return torch.matmul(T_t, S_t).tolist()
''',

    "93-Calculate Mean Absolute Error (MAE).py": '''import torch
import torch.nn.functional as F

def mae(y_true: list[float], y_pred: list[float]) -> float:
    t_true = torch.tensor(y_true, dtype=torch.float32)
    t_pred = torch.tensor(y_pred, dtype=torch.float32)
    return F.l1_loss(t_pred, t_true).item()
''',

    "94-Implement Multi-Head Attention.py": '''import torch
import torch.nn as nn

def mha_forward(q: list[list[float]], k: list[list[float]], v: list[list[float]], num_heads: int) -> list[list[float]]:
    """
    使用 PyTorch 官方 nn.MultiheadAttention 实现
    """
    Q = torch.tensor(q, dtype=torch.float32).unsqueeze(1) # (seq, batch, dim)
    K = torch.tensor(k, dtype=torch.float32).unsqueeze(1)
    V = torch.tensor(v, dtype=torch.float32).unsqueeze(1)
    
    embed_dim = Q.shape[-1]
    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=False)
    
    attn_output, _ = mha(Q, K, V)
    return attn_output.squeeze(1).tolist()
''',

    "96-Singular Value Decomposition (SVD) of 2x2 Matrix.py": '''import torch

def svd_2x2(matrix: list[list[float]]) -> tuple:
    mat_t = torch.tensor(matrix, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(mat_t)
    return U.tolist(), S.tolist(), Vh.tolist()
''',

    "97-Determinant of a 4x4 Matrix using Laplace's Expansion.py": '''import torch

def determinant_4x4(matrix: list[list[float]]) -> float:
    mat_t = torch.tensor(matrix, dtype=torch.float32)
    return torch.linalg.det(mat_t).item()
''',

    "98-Overlapping Max Pooling.py": '''import torch
import torch.nn as nn

def overlapping_max_pooling(image: list[list[float]], kernel_size: int, stride: int) -> list[list[float]]:
    # shape: (batch, channels, height, width)
    t = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    return pool(t).squeeze(0).squeeze(0).tolist()
''',

    "99-Implement Local Response Normalization.py": '''import torch
import torch.nn as nn

def local_response_normalization(feature_map: list[list[list[float]]], size: int, alpha: float, beta: float, k: float) -> list[list[list[float]]]:
    # shape (batch, channels, height, width)
    t = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0)
    lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    return lrn(t).squeeze(0).tolist()
''',

    "101-Dropout Layer.py": '''import torch
import torch.nn as nn

def dropout_forward(features: list[float], p: float, training: bool) -> list[float]:
    t = torch.tensor(features, dtype=torch.float32)
    dropout = nn.Dropout(p=p)
    if training:
        dropout.train()
    else:
        dropout.eval()
    return dropout(t).tolist()
''',

    "103-Implement Global Average Pooling.py": '''import torch
import torch.nn as nn

def global_average_pooling(feature_map: list[list[list[float]]]) -> list[float]:
    # (batch, channels, h, w)
    t = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0)
    gap = nn.AdaptiveAvgPool2d(1)
    return gap(t).squeeze(-1).squeeze(-1).squeeze(0).tolist()
''',

    "104-Batch Normalization for BCHW Input.py": '''import torch
import torch.nn as nn

def batch_normalization_bchw(feature_map: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    t = torch.tensor(feature_map, dtype=torch.float32)
    channels = t.shape[1]
    bn = nn.BatchNorm2d(channels)
    return bn(t).tolist()
''',

    "106-Implement Group Normalization.py": '''import torch
import torch.nn as nn

def group_normalization(feature_map: list[list[list[list[float]]]], num_groups: int) -> list[list[list[list[float]]]]:
    t = torch.tensor(feature_map, dtype=torch.float32)
    channels = t.shape[1]
    gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    return gn(t).tolist()
'''
}

out_dir = "pytorch"
import os
os.makedirs(out_dir, exist_ok=True)

for name, content in files_content.items():
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(content)

print("Batch 6 & 7 PyTorch completed!")
