import torch
import torch.nn as nn
import torch.nn.functional as F

class XXNet(nn.Module):
    def __init__(self):
        super(XXNet, self).__init__()

        self.fc1 = nn.Linear(8, 1)

    def forward(self, data):
        net = self.fc1(data)
        return net


