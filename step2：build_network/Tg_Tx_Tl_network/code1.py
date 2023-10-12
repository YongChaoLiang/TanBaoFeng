'''
搭建ResNet神经网络，
网络架构：ResNet10
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
# from torchviz import make_dot


class Ann(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Ann, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    temp = torch.randn(10, 25)
    net = Ann(in_dim=25,out_dim=3)
    out = net(temp)

    g = make_dot(out)

    g.render('espnet_model', view=False)