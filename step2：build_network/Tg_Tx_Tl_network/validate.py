import torch
import numpy as np
from code1 import Ann
from sklearn import metrics
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

torch.set_default_tensor_type(torch.DoubleTensor)


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='12', weight='bold')  # 设置字体样式、大小
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

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes[::-1],
           title=title,
           ylabel='Actual Class',
           xlabel='Predicted Class with ANN')

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
                        color="white" if cm[i, j] > thresh else "black", size=14)
    fig.tight_layout()
    # plt.savefig('cm.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    data = np.load('data/scale_data.npy')
    true_value = np.load('data/value.npy')

    data = torch.from_numpy(data)

    network = Ann(in_dim=25, out_dim=3)

    files = os.listdir('model')

    network.load_state_dict(torch.load('./model/{}'.format('t.pt')))

    out = network(data)
    out = out.detach().numpy()
    out = np.exp(out)

    rmse = np.sqrt(mean_squared_error(true_value[:, 2], out[:, 2]))

    res = pd.DataFrame(columns=['true', 'pred'])

    # title = ['Tg', 'Tx', 'Tl']
    # for i in range(3):
    #     res['true'] = true_value[:, i]
    #     res['pred'] = out[:, i]
    #     writer = pd.ExcelWriter('{}_true_pred.xlsx'.format(title[i]))
    #     res.to_excel(writer,index=False)
    #     writer.save()

    title = ['Tg', 'Tx', 'Tl']
    for i in range(3):
        plt.subplot(2, 2, i + 2)
        res['true'] = true_value[:, i]
        res['pred'] = out[:, i]
        rmse = np.round(np.sqrt(mean_squared_error(np.log(res['true']), np.log(res['pred']))), 3)
        g = sns.regplot('true', "pred", res)
        pearson = round(res['true'].corr(res['pred']), 2)
        plt.text(0.3, 0.9, '{}_rmse: {}'.format(title[i], rmse), transform=g.transAxes, size=10)
        plt.tight_layout()

    # plt.subplot(2,2,1)
    # dmax_res = pd.read_excel('true_pred.xlsx')
    # g = sns.regplot('true', "pred", dmax_res)
    # pearson = round(dmax_res['true'].corr(dmax_res['pred']), 6)
    # plt.text(0.3, 0.8, 'mse: {} \npearson: {}'.format(mse, pearson), size=14, transform=g.transAxes)
    # plt.tight_layout()
    #
    plt.show()
