import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, output, target):
        m = nn.LogSoftmax(dim=1)
        return self.loss(m(output), target)
