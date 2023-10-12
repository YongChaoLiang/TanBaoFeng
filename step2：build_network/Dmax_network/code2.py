"""
构建数据集，加载数据
"""
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DmaxDataset(Dataset):
    def __init__(self, data, value, train):
        super(DmaxDataset, self).__init__()
        self.data = data
        self.value = value
        self.train = train

        # 测试集索引
        test_idx = [i for i in range(0, len(self.data), 4)]
        # 训练集索引
        train_idx = list(set([i for i in range(len(self.data))]) - set(test_idx))
        
        # 训练集数据
        if self.train:
            self.data = self.data[train_idx]
            self.value = self.value[train_idx]
            
        # 测试集数据
        else:
            self.data = self.data[test_idx]
            self.value = self.value[test_idx]

    def __getitem__(self, idx):

        value = self.value[idx]
        data = self.data[idx]

        return data, value    # 返回你要提供给Dataloader的一个样本（数据+标签）

    def __len__(self):
        return len(self.value)
