import os

files_content = {
    "5-Scalar Multiplication of a Matrix.py": '''import torch

def scalar_multiply(matrix: list[list[float]], scalar: float) -> list[list[float]]:
    """
    用 PyTorch 实现矩阵的标量乘法
    """
    mat_t = torch.tensor(matrix, dtype=torch.float32)
    result = mat_t * scalar
    return result.tolist()
''',

    "6-Calculate Eigenvalues of a Matrix.py": '''import torch

def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
    """
    用 PyTorch 计算矩阵的特征值
    """
    mat_t = torch.tensor(matrix, dtype=torch.float32)
    # torch.linalg.eigvals 返回复数 tensor
    eigvals = torch.linalg.eigvals(mat_t)
    return eigvals.real.tolist()
''',

    "8-Calculate 2x2 Matrix Inverse.py": '''import torch

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    """
    用 PyTorch 计算矩阵逆
    """
    mat_t = torch.tensor(matrix, dtype=torch.float32)
    inv_t = torch.linalg.inv(mat_t)
    return inv_t.tolist()
''',

    "9-Matrix times Matrix.py": '''import torch

def matrixmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """
    用 PyTorch 实现矩阵乘法
    """
    mat_a = torch.tensor(a, dtype=torch.float32)
    mat_b = torch.tensor(b, dtype=torch.float32)
    if mat_a.shape[1] != mat_b.shape[0]:
        return -1
    return torch.matmul(mat_a, mat_b).tolist()
''',

    "10-Poisson Distribution Probability Calculator.py": '''import torch
from torch.distributions import Poisson

def poisson_probability(lam: float, k: int) -> float:
    """
    用 PyTorch distributions 计算泊松分布概率
    """
    dist = Poisson(torch.tensor(lam, dtype=torch.float32))
    # 取对数概率再指数化 (也可直接用其他公式)
    return dist.log_prob(torch.tensor(k, dtype=torch.float32)).exp().item()
''',

    "11-Binomial Distribution Probability.py": '''import torch
from torch.distributions import Binomial

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    用 PyTorch 计算二项分布概率
    """
    dist = Binomial(total_count=n, probs=torch.tensor(p, dtype=torch.float32))
    return dist.log_prob(torch.tensor(k, dtype=torch.float32)).exp().item()
''',

    "12-Normal Distribution PDF Calculator.py": '''import torch
from torch.distributions import Normal

def normal_pdf(x: float, mean: float, std: float) -> float:
    """
    用 PyTorch 计算正态分布的概率密度函数 (PDF)
    """
    dist = Normal(torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32))
    return dist.log_prob(torch.tensor(x, dtype=torch.float32)).exp().item()
''',

    "13-Descriptive Statistics Calculator.py": '''import torch

def descriptive_statistics(data: list[float]) -> dict:
    """
    用 PyTorch 计算均值、方差和标准差
    """
    t = torch.tensor(data, dtype=torch.float32)
    return {
        "mean": t.mean().item(),
        "variance": t.var(unbiased=True).item(),  # 样本方差
        "std": t.std(unbiased=True).item()        # 样本标准差
    }
''',

    "14-Calculate Covariance Matrix.py": '''import torch

def covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """
    用 PyTorch 计算协方差矩阵
    """
    t = torch.tensor(vectors, dtype=torch.float32)
    # 按照 Numpy 和 PyTorch cov 的习惯，每一行代表一个变量
    cov = torch.cov(t)
    return cov.tolist()
''',

    "15-Linear Regression Using Gradient Descent.py": '''import torch

def linear_regression_gd(X: list[list[float]], y: list[float], lr: float, epochs: int) -> tuple:
    """
    使用 PyTorch 自带的 Autograd (自动求导) 来实现梯度下降
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # 定义需要梯度的参数
    weights = torch.zeros((X_t.shape[1], 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    
    # 我们可以直接使用手动更新，或者用 SGD 优化器
    optimizer = torch.optim.SGD([weights, bias], lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = X_t @ weights + bias
        loss = loss_fn(predictions, y_t)
        loss.backward()
        optimizer.step()
        
    return weights.detach().flatten().tolist(), bias.item()
'''
}

out_dir = "pytorch"
os.makedirs(out_dir, exist_ok=True)

for name, content in files_content.items():
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(content)

print("Batch 1 PyTorch completed!")
