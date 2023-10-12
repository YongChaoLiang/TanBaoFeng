"""
主运行程序
"""
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from code2 import TempDataset
from code1 import Ann
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(3)


def main():
    temp_data = np.load('data/scale_data.npy')
    temp_value = np.load('data/log_value.npy')
    writer = SummaryWriter('./logs')

    # 打乱数据和标签，保证数据和标签对应一致
    state = np.random.get_state()
    np.random.shuffle(temp_data)
    np.random.set_state(state)
    np.random.shuffle(temp_value)

    batch_size = 32

    train_data = TempDataset(temp_data, temp_value, train=True)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = TempDataset(temp_data, temp_value, train=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')

    network = Ann(in_dim=25, out_dim=3)
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        network = network.to(device)
        loss_fn = loss_fn.to(device)

    epoch = 1000
    best_model = 0
    for i in range(epoch):

        network.train()
        total_train_loss = 0
        for each in train_data_loader:

            data, target = each
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            # print(data.shape)
            outputs = network(data)
            loss = loss_fn(outputs, target)
            total_train_loss += loss.item()
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 进行优化
            optimizer.step()

        print("训练次数：{}, Loss: {}".format(i + 1, total_train_loss / len(train_data_loader)))
        writer.add_scalar('train_loss', total_train_loss / len(train_data_loader), i + 1)

        # 进行测试
        network.eval()
        total_test_loss = 0
        total_acc = 0
        with torch.no_grad():
            for each in test_data_loader:

                data, target = each
                if torch.cuda.is_available():
                    data = data.to(device)
                    target = target.to(device)

                outputs = network(data)
                loss = loss_fn(outputs, target)
                total_test_loss += loss.item()

        test_loss = total_test_loss / len(test_data_loader)
        print("测试集Loss: {}".format(test_loss))
        writer.add_scalar('test_loss', test_loss, i + 1)

        torch.save(network.state_dict(), './model3/第{}轮.{}.pt'.format(i + 1, np.round(test_loss, 6)))

    writer.close()


if __name__ == '__main__':
    main()
