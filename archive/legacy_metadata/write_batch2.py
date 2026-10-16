import os

files_content = {
    "16-Implement Gradient Descent Variants with MSE Loss.py": '''import torch

def gradient_descent_variants(X: list[list[float]], y: list[float], lr: float, epochs: int, momentum: float = 0.0) -> tuple:
    """
    使用 PyTorch 演示带动量的梯度下降变体
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    weights = torch.zeros((X_t.shape[1], 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    
    # 包含了标准 SGD 和带 Momentum 的 SGD
    optimizer = torch.optim.SGD([weights, bias], lr=lr, momentum=momentum)
    loss_fn = torch.nn.MSELoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(X_t @ weights + bias, y_t)
        loss.backward()
        optimizer.step()
        
    return weights.detach().flatten().tolist(), bias.item()
''',

    "17-Implement Adam Optimization Algorithm.py": '''import torch

def adam_optimizer(X: list[list[float]], y: list[float], lr: float, epochs: int) -> tuple:
    """
    使用 PyTorch 内置的 Adam 优化器
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    weights = torch.zeros((X_t.shape[1], 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    
    optimizer = torch.optim.Adam([weights, bias], lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(X_t @ weights + bias, y_t)
        loss.backward()
        optimizer.step()
        
    return weights.detach().flatten().tolist(), bias.item()
''',

    "17-K-Means Clustering.py": '''import torch

def k_means_clustering(data: list[list[float]], k: int, max_iters: int = 100) -> list[int]:
    """
    使用 PyTorch 实现 K-Means 聚类
    """
    X = torch.tensor(data, dtype=torch.float32)
    N, D = X.shape
    
    # 随机初始化聚类中心 (也可以选择前 k 个)
    centroids = X[:k].clone()
    
    cluster_assignments = torch.zeros(N, dtype=torch.long)
    
    for _ in range(max_iters):
        # 计算距离矩阵 (N, k)
        distances = torch.cdist(X, centroids)
        # 找到最近的中心点
        new_assignments = torch.argmin(distances, dim=1)
        
        if torch.equal(new_assignments, cluster_assignments):
            break
            
        cluster_assignments = new_assignments
        
        # 更新聚类中心
        for i in range(k):
            mask = (cluster_assignments == i)
            if mask.sum() > 0:
                centroids[i] = X[mask].mean(dim=0)
                
    return cluster_assignments.tolist()
''',

    "18-Implement K-Fold Cross-Validation.py": '''import torch

def k_fold_cross_validation(data: list[list[float]], k: int) -> list[tuple]:
    """
    使用 PyTorch 实现 K 折交叉验证的数据划分
    """
    X = torch.tensor(data, dtype=torch.float32)
    N = X.shape[0]
    
    # 获取随机排列的索引
    indices = torch.randperm(N)
    fold_size = N // k
    
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else N
        
        test_indices = indices[start:end]
        train_indices = torch.cat((indices[:start], indices[end:]))
        
        folds.append((train_indices.tolist(), test_indices.tolist()))
        
    return folds
''',

    "18-Implement Lasso Regression using ISTA.py": '''import torch

def lasso_ista(X: list[list[float]], y: list[float], alpha: float, lr: float, epochs: int) -> list[float]:
    """
    使用 PyTorch 实现带有 L1 正则化的 ISTA 算法 (Lasso 回归)
    软阈值 (Soft Thresholding) 实现
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    w = torch.zeros((X_t.shape[1], 1))
    
    for _ in range(epochs):
        # 梯度步
        pred = X_t @ w
        grad = X_t.T @ (pred - y_t) / len(y_t)
        w = w - lr * grad
        
        # 近端步 (Proximal step): 软阈值
        w = torch.sign(w) * torch.clamp(torch.abs(w) - lr * alpha, min=0.0)
        
    return w.flatten().tolist()
''',

    "19-Principal Component Analysis (PCA) Implementation.py": '''import torch

def pca_implementation(X: list[list[float]], n_components: int) -> list[list[float]]:
    """
    使用 PyTorch 实现主成分分析 (PCA)
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    
    # 1. 中心化
    X_centered = X_t - X_t.mean(dim=0)
    
    # 2. SVD 分解。PyTorch 也有 torch.pca_lowrank，这里用 SVD 更直观
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # 3. 取前 n_components 个主成分
    V_subset = Vh.T[:, :n_components]
    
    # 4. 投影到主成分空间
    X_reduced = torch.matmul(X_centered, V_subset)
    return X_reduced.tolist()
''',

    "19-Softmax Activation Function Implementation.py": '''import torch
import torch.nn.functional as F

def softmax(x: list[float]) -> list[float]:
    """
    使用 PyTorch 的 F.softmax
    """
    t = torch.tensor(x, dtype=torch.float32)
    return F.softmax(t, dim=-1).tolist()
''',

    "20-Implementation of Log Softmax Function.py": '''import torch
import torch.nn.functional as F

def log_softmax(x: list[float]) -> list[float]:
    """
    使用 PyTorch 的 F.log_softmax
    """
    t = torch.tensor(x, dtype=torch.float32)
    return F.log_softmax(t, dim=-1).tolist()
''',

    "21-Sigmoid Activation Function Understanding.py": '''import torch

def sigmoid_activation(x: list[float]) -> list[float]:
    """
    使用 PyTorch 的 torch.sigmoid
    """
    t = torch.tensor(x, dtype=torch.float32)
    return torch.sigmoid(t).tolist()
''',

    "22-Implement ReLU Activation Function.py": '''import torch
import torch.nn.functional as F

def relu_activation(x: list[float]) -> list[float]:
    """
    使用 PyTorch 的 F.relu
    """
    t = torch.tensor(x, dtype=torch.float32)
    return F.relu(t).tolist()
''',

    "23-Leaky ReLU Activation Function.py": '''import torch
import torch.nn.functional as F

def leaky_relu(x: list[float], negative_slope: float = 0.01) -> list[float]:
    """
    使用 PyTorch 的 F.leaky_relu
    """
    t = torch.tensor(x, dtype=torch.float32)
    return F.leaky_relu(t, negative_slope=negative_slope).tolist()
''',

    "24-Implement the PReLU Activation Function.py": '''import torch
import torch.nn as nn

def prelu_activation(x: list[float], init: float = 0.25) -> list[float]:
    """
    使用 PyTorch 的 PReLU
    """
    t = torch.tensor(x, dtype=torch.float32)
    # PReLU 包含可学习参数，这里展示如何前向计算
    prelu = nn.PReLU(init=init)
    return prelu(t).tolist()
''',

    "25-Single Neuron.py": '''import torch
import torch.nn as nn

def single_neuron_forward(features: list[float], weights: list[float], bias: float) -> float:
    """
    单神经元前向传播
    """
    x = torch.tensor(features, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    b = torch.tensor(bias, dtype=torch.float32)
    
    # Dot product + bias, then sigmoid
    z = torch.dot(x, w) + b
    return torch.sigmoid(z).item()
''',

    "25-Single Neuron with Backpropagation.py": '''import torch

def single_neuron_backprop(features: list[float], weights: list[float], bias: float, target: float, lr: float) -> tuple:
    """
    单神经元反向传播 (利用 PyTorch Autograd)
    """
    x = torch.tensor(features, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(bias, dtype=torch.float32, requires_grad=True)
    y_true = torch.tensor(target, dtype=torch.float32)
    
    optimizer = torch.optim.SGD([w, b], lr=lr)
    
    # Forward
    y_pred = torch.sigmoid(torch.dot(x, w) + b)
    
    # MSE Loss
    loss = (y_pred - y_true)**2
    
    # Backward
    loss.backward()
    optimizer.step()
    
    return w.detach().tolist(), b.item()
''',

    "26-Implementing Basic Autograd Operations.py": '''import torch

def basic_autograd(x_val: float) -> float:
    """
    演示基本 Autograd: 给定 y = x^2 + 2x，求 dy/dx
    """
    x = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    y = x**2 + 2*x
    y.backward()
    return x.grad.item()
''',

    "26-Implementing a Simple RNN.py": '''import torch
import torch.nn as nn

def simple_rnn_forward(seq: list[list[float]], input_size: int, hidden_size: int) -> list[list[float]]:
    """
    使用 PyTorch 官方 nn.RNN 实现
    """
    # 假设 seq shape: (seq_len, batch_size, input_size)，这里简化为无 batch (seq_len, input_size)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(1) # (seq_len, 1, input_size)
    
    rnn = nn.RNN(input_size, hidden_size, batch_first=False)
    # 初始化权重为 0 或固定值仅为测试，这里直接用默认初始化
    out, h_n = rnn(x)
    return out.squeeze(1).tolist()
''',

    "27-Implement Long Short-Term Memory (LSTM) Network.py": '''import torch
import torch.nn as nn

def lstm_forward(seq: list[list[float]], input_size: int, hidden_size: int) -> list[list[float]]:
    """
    使用 PyTorch 官方 nn.LSTM 实现
    """
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(1) # (seq_len, 1, input_size)
    
    lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
    out, (h_n, c_n) = lstm(x)
    return out.squeeze(1).tolist()
'''
}

out_dir = "pytorch"
import os
os.makedirs(out_dir, exist_ok=True)

for name, content in files_content.items():
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(content)

print("Batch 2 & 3 PyTorch completed!")
