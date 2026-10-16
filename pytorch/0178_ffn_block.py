"""Deep-ML #178: Implement Position-wise Feed-Forward Block with Residual and Dropout.
https://www.deep-ml.com/problems/178
Standalone PyTorch solution; public-example replay is not an official submission.
The PyTorch judge interface is FFNBlock, not the NumPy ffn function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNBlock(nn.Module):
    def __init__(self, d_model, d_hidden, dropout_p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        return x + self.dropout(self.linear2(hidden))
