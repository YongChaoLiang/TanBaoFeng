import torch
import torch.nn as nn
from torch.nn import functional as F
from torchviz import make_dot

class Ann(nn.Module):
    def __init__(self):
        super(Ann, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(25, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    temp = torch.randn(3, 25)
    net = Ann()
    out = net(temp)
    g = make_dot(out)

    g.render('espnet_model', view=False)
