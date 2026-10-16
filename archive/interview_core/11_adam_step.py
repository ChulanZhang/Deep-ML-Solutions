import torch

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam 优化器单步更新 - 经典优化算法手写题
    核心考点:
    1. 一阶动量 m (梯度期望均值): 结合历史梯度方向，冲过鞍点
    2. 二阶动量 v (梯度平方均值): 结合历史梯度尺度，自适应调节每个参数的学习率
    3. 偏差修正 (Bias Correction): 修正初始阶段 m=0, v=0 引起的偏向 0 估计
       m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    4. 参数更新: param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """
    # 1. 更新一阶和二阶动量
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # 2. 偏差修正 (随步数 t 增加，1 - beta^t 逐渐趋向于 1)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    # 3. 梯度下降更新参数
    param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)

    return param, m, v

if __name__ == "__main__":
    w = torch.tensor([1.0, 2.0])
    g = torch.tensor([0.1, -0.2])
    m = torch.zeros_like(w)
    v = torch.zeros_like(w)
    
    # 模拟第 1 步
    w, m, v = adam_step(w, g, m, v, t=1)
    print("Adam Step 1 Updated Param:", w)
