import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from code2 import DmaxDataset
from code1 import Ann
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(4)


def main():
    dmax_data = np.load('data/scale_data.npy')
    dmax_value = np.load('data/log_value.npy')
    writer = SummaryWriter('./logs')

    batch_size = 128

    train_data = DmaxDataset(dmax_data, dmax_value, train=True)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = DmaxDataset(dmax_data, dmax_value, train=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')

    network = Ann()
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        network = network.to(device)
        loss_fn = loss_fn.to(device)

    epoch = 1000
    best_model = np.Inf
    best_time = 0


    # train
    for i in range(epoch):
        network.train()
        total_train_loss = 0
        for each in train_data_loader:

            data, value = each
            if torch.cuda.is_available():
                data = data.to(device)
                value = value.to(device)

            # print(data.shape)
            outputs = network(data)
            loss = loss_fn(outputs.view(-1), value)
            total_train_loss += loss.item()
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 进行优化，更新权重参数
            optimizer.step()

        train_loss = total_train_loss / len(train_data_loader)
        print("训练次数：{}, Loss: {}".format(i + 1, train_loss))
        writer.add_scalar('train_loss', train_loss, i + 1)

        # 进行测试
        network.eval()
        total_test_loss = 0
        with torch.no_grad():
            for each in test_data_loader:
                data, value = each
                if torch.cuda.is_available():
                    data = data.to(device)
                    value = value.to(device)

                outputs = network(data)
                loss = loss_fn(outputs.view(-1), value)
                total_test_loss += loss.item()

        test_loss = total_test_loss / len(test_data_loader)
        print("测试集Loss: {}".format(test_loss))
        writer.add_scalar('test_loss', test_loss, i + 1)

        # 存储每一轮模型
        # torch.save(network.state_dict(), './model/第{}轮.{}.pt'.format(i + 1, np.round(test_loss, 6)))
        if  test_loss < best_model:
            best_time = i + 1
            best_model = test_loss
            torch.save(network.state_dict(), './model/model_{}.pt'.format(i+1))

    print("最佳模型为第{}轮模型,test_loss为{:.6f}".format(best_time, best_model))

    writer.close()


if __name__ == '__main__':
    main()
