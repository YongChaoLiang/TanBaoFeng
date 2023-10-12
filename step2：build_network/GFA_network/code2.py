"""
加载数据，进行训练
"""
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torchvision import transforms


class GfaDataset(Dataset):
    def __init__(self, data, label, train):
        super(GfaDataset, self).__init__()
        self.data = data
        self.label = label
        self.train = train

        if self.train:
            self.data = np.asarray(list(self.data)[:1344] +
                                   list(self.data)[1920:4935] +
                                   list(self.data)[6228:7325])
            self.label = np.asarray(list(self.label)[:1344] +
                                    list(self.label)[1920:4935] +
                                    list(self.label)[6228:7325])
        else:
            self.data = np.asarray(list(self.data)[1344:1920] +
                                   list(self.data)[4935:6228] +
                                   list(self.data)[7325:])
            self.label = np.asarray(list(self.label)[1344:1920] +
                                    list(self.label)[4935:6228] +
                                    list(self.label)[7325:])

    def __getitem__(self, idx):

        label = self.label[idx]

        label = torch.zeros(3).scatter(0, torch.LongTensor([label]), 1)

        data = self.data[idx]

        return data, label


    def __len__(self):
        return len(self.data)