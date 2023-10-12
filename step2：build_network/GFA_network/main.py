"""
主运行程序
"""
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from code2 import GfaDataset
from code1 import Ann
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(1)


def main():
    gfa_data = np.load('data/scale_data.npy')
    gfa_data = gfa_data.reshape((len(gfa_data), -1))
    gfa_label = np.load('data/label.npy', allow_pickle=True)
    writer = SummaryWriter('./logs')

    # 打乱数据和标签，保证数据和标签对应一致
    state = np.random.get_state()


    np.random.shuffle(gfa_data[:1920])
    np.random.shuffle(gfa_data[1920:6228])
    np.random.shuffle(gfa_data[6228:])
    np.random.set_state(state)
    np.random.shuffle(gfa_label[:1920])
    np.random.shuffle(gfa_label[1920:6228])
    np.random.shuffle(gfa_label[6228:])

    batch_size = 64

    train_data = GfaDataset(gfa_data, gfa_label, train=True)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = GfaDataset(gfa_data, gfa_label, train=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')

    network = Ann(in_dim=25, out_dim=3)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        network = network.to(device)
        loss_fn = loss_fn.to(device)

    epoch = 1500
    best_model = 0
    for i in range(epoch):
        # 训练
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

        # 测试
        network.eval()
        total_test_loss = 0
        total_acc = 0
        with torch.no_grad():   # 测试部分无需反向传播
            for each in test_data_loader:

                data, target = each
                if torch.cuda.is_available():
                    data = data.to(device)
                    target = target.to(device)

                outputs = network(data)
                loss = loss_fn(outputs, target)
                total_test_loss += loss.item()
                acc = (outputs.argmax(1) == target.argmax(1)).cpu().sum().numpy()
                total_acc += acc

        # print("测试集Loss: {}".format(total_test_loss))
        writer.add_scalar('test_loss', total_test_loss / len(test_data_loader), i + 1)

        model_acc = total_acc / len(test_data)
        print('acc：{}'.format(model_acc))
        writer.add_scalar('acc', model_acc, i + 1)

        # 保存模型
        if i % 100 == 0:
            torch.save(network.state_dict(), './model3/第{}轮.{}.pt'.format(i + 1, np.round(model_acc, 3)))

        if model_acc > best_model:
            best_model = model_acc
            torch.save(network.state_dict(), './model3/best_model/第{}轮.{}.pt'.format(i + 1, np.round(best_model, 3)))

    writer.close()


if __name__ == '__main__':
    main()
