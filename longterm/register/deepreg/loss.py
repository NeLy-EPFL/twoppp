# Jonas Braun
# jonas.braun@epfl.ch
# 22.02.2021

# copied from Semih Günel's repo https://github.com/NeLy-EPFL/Drosoph2PRegistration

import torch
import torch.nn as nn
from torch.autograd import Variable


class TotalVaryLoss(nn.Module):
    def __init__(self):
        super(TotalVaryLoss, self).__init__()

    def forward(self, x, weight=1e-6):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (
            torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
            + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )
        return self.loss


class BiasReduceLoss(nn.Module):
    def __init__(self):
        super(BiasReduceLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, y, weight=1e-2):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x, 0).unsqueeze(0)
        self.loss = w * self.criterion(self.avg, y)
        return self.loss
