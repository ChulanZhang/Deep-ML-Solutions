import torch

def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    数值稳定版 Softmax - 面试必考数值稳定性
    核心考点:
    直接算 exp(x) 在输入大于 88 时(FP32)会发生数值溢出 (Inf)
    技巧: 减去沿维度的最大值 x_max，使得最大值为 0，exp 范围落在 (0, 1] 之间，绝不溢出
    """
    max_val = torch.amax(logits, dim=dim, keepdim=True)
    exp_logits = torch.exp(logits - max_val)
    return exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)

def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    数值稳定版交叉熵损失 (Log-Sum-Exp 技巧)
    公式: Loss = -log(softmax(logits)[target])
         = -(logits[target] - log(sum(exp(logits))))
         = log_sum_exp(logits) - logits[target]
    """
    # 1. 计算 Log-Sum-Exp: m + log(sum(exp(x - m)))
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    lse = max_logits + torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1, keepdim=True))
    
    # 2. 提取真实类别对应的 logit: logits.gather(1, targets)
    target_logits = logits.gather(dim=1, index=targets.unsqueeze(1))
    
    # 3. 交叉熵损失 = LSE - target_logits 的样本均值
    loss = (lse - target_logits).mean()
    return loss

if __name__ == "__main__":
    # 极端大数测试: 普通 exp 会产生 Inf
    logits = torch.tensor([[1000.0, 1001.0, 1002.0], [500.0, 501.0, 502.0]])
    targets = torch.tensor([2, 1])

    probs = stable_softmax(logits)
    print("Stable Softmax 输出无 Inf/NaN:\n", probs)
    loss = stable_cross_entropy(logits, targets)
    print("Stable CrossEntropy Loss:", loss.item())
