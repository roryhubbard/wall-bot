import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.fl1 = nn.Linear(2, 32)
        self.fl2 = nn.Linear(32, n_actions)

    def forward(self, x):
        h = F.relu(self.fl1(x))
        h = self.fl2(h)
        return h
