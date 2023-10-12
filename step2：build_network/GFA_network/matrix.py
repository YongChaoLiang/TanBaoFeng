"""
绘制混淆矩阵
"""
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from code1 import Ann
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='20')  # 设置字体样式、大小
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, 6),dpi=200)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes[::-1],
           title=title,
           ylabel='Actual Class',
           xlabel='Predicted Class with ANN    (a)')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", size=17, weight='bold')
    fig.tight_layout()
    # plt.savefig('cm.jpg', dpi=300)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    network = Ann(in_dim=25, out_dim=3)
    # params = torch.load('model/best_model/第646轮.0.902.pt')
    params = torch.load('model3/best_model/第300轮.0.898.pt')
    network.load_state_dict(params)

    gfa_data = np.load('data/scale_data.npy')
    gfa_data = gfa_data.reshape((len(gfa_data), -1))
    gfa_data = torch.from_numpy(gfa_data)
    gfa_label = np.load('data/label.npy')

    outputs = network(gfa_data)
    pred = outputs.detach().numpy().argmax(1)

    matrix = confusion_matrix(gfa_label, pred)
    matrix = np.flip(matrix, axis=0)
    print(matrix)

    plot_Matrix(matrix, classes=['CRA', 'RMG', 'BMG'])
