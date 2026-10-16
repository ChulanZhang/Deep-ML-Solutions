"""Deep-ML #382: Direct Preference Optimization (DPO) Loss.
https://www.deep-ml.com/problems/382
Standalone PyTorch solution; public-example replay is not an official submission.
Official output is a rounded dict; interview_core provides a differentiable Tensor loss.
"""
import torch
import torch.nn.functional as F

def dpo_loss(log_probs_chosen_policy, log_probs_rejected_policy,
             log_probs_chosen_ref, log_probs_rejected_ref, beta):
    pc, pr, rc, rr = (torch.as_tensor(v, dtype=torch.float64) for v in
                      (log_probs_chosen_policy, log_probs_rejected_policy,
                       log_probs_chosen_ref, log_probs_rejected_ref))
    chosen, rejected = beta * (pc - rc), beta * (pr - rr)
    loss = -F.logsigmoid(chosen - rejected).mean()
    return {'loss': round(loss.item(), 4),
            'chosen_rewards': [round(v, 4) for v in chosen.tolist()],
            'rejected_rewards': [round(v, 4) for v in rejected.tolist()]}
