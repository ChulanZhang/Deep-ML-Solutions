import torch

def mlp_forward_backward(X, Y, W1, b1, W2, b2):
    """
    两层 MLP 手写反向传播 (纯矩阵求导与链式法则，不使用 autograd)
    前向: 
       Z1 = X @ W1 + b1
       A1 = relu(Z1)
       Z2 = A1 @ W2 + b2 (预测值 Y_hat)
       Loss = 0.5 * mean((Z2 - Y)^2)
    核心考点:
    1. dZ2 = (Z2 - Y) / B
    2. dW2 = A1.T @ dZ2,  db2 = sum(dZ2, dim=0)
    3. dA1 = dZ2 @ W2.T
    4. dZ1 = dA1 * (Z1 > 0)  [ReLU 导数]
    5. dW1 = X.T @ dZ1,  db1 = sum(dZ1, dim=0)
    """
    B = X.shape[0]

    # --- 1. 前向传播 ---
    Z1 = X @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2  # Y_hat
    loss = 0.5 * torch.mean((Z2 - Y) ** 2)

    # --- 2. 反向传播 (链式法则) ---
    # dLoss / dZ2
    dZ2 = (Z2 - Y) / (B * Z2.shape[1])
    
    # 对 W2, b2 求导
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(dim=0)

    # 传回隐藏层激活值
    dA1 = dZ2 @ W2.T
    
    # 经过 ReLU 的导数 (Z1 > 0 处为 1，其余为 0)
    dZ1 = dA1 * (Z1 > 0).float()
    
    # 对 W1, b1 求导
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(dim=0)

    return loss, (dW1, db1, dW2, db2)

if __name__ == "__main__":
    X = torch.randn(4, 3)
    Y = torch.randn(4, 2)
    W1 = torch.randn(3, 5)
    b1 = torch.zeros(5)
    W2 = torch.randn(5, 2)
    b2 = torch.zeros(2)

    loss, grads = mlp_forward_backward(X, Y, W1, b1, W2, b2)
    print("Forward Loss:", loss.item())
    print("Gradients computed successfully: dW1 shape =", grads[0].shape)
