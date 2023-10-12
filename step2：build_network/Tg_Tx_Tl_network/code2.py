"""
加载数据，进行训练
"""
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torchvision import transforms


class TempDataset(Dataset):
    def __init__(self, data, value, train):
        super(TempDataset, self).__init__()
        self.data = data
        self.value = value
        self.train = train

        if self.train:
            self.data = self.data[:int(0.7 * len(self.data))]
            self.value = self.value[:int(0.7 * len(self.value))]
        else:
            self.data = self.data[int(0.7 * len(self.data)):]
            self.value = self.value[int(0.7 * len(self.value)):]

    def __getitem__(self, idx):

        value = self.value[idx]

        data = self.data[idx]

        return data, value

    def __len__(self):
        return len(self.data)
