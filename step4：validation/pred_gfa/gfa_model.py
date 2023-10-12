'''
搭建ResNet神经网络，
网络架构：ResNet10
'''
import torch

from torchviz import make_dot
import torch.nn as nn


class Ann(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Ann, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    temp = torch.randn(1, 25)
    net = Ann(in_dim=25,out_dim=3)
    out = net(temp)

    g = make_dot(out)

    g.render('espnet_model', view=False)
