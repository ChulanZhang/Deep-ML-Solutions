import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    直接偏好优化损失 (DPO Loss) - 大模型后训练对齐 (RLHF) 最热考点
    核心考点:
    1. 彻底跳过奖励模型 (Reward Model) 和 PPO 强化学习，用解析解直接优化策略
    2. 隐式隐式奖励差公式:
       r_diff = beta * [(policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)]
    3. 损失函数: -log(sigmoid(r_diff)) -> 采用 F.logsigmoid 避免数值下溢
    """
    # 1. 计算策略模型相对于参考模型的隐式优势 log ratio
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # 2. 隐式奖励之差
    logits = beta * (chosen_logratios - rejected_logratios)

    # 3. 优化目标: 最大化偏好回答概率，最小化劣质回答概率
    losses = -F.logsigmoid(logits)
    return losses.mean()

if __name__ == "__main__":
    # 模拟 2 个样本在策略模型和参考模型下的序列对数概率和
    p_w = torch.tensor([-2.1, -1.8])
    p_l = torch.tensor([-4.5, -3.9])
    ref_w = torch.tensor([-2.5, -2.0])
    ref_l = torch.tensor([-3.0, -2.8])

    loss = dpo_loss(p_w, p_l, ref_w, ref_l, beta=0.1)
    print("DPO Loss:", loss.item())
