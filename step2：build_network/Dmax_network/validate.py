import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from code1 import Ann
import torch
from sklearn.metrics import mean_squared_error  # 均方误差

torch.set_default_tensor_type(torch.DoubleTensor)
mpl.use('TkAgg')

if __name__ == '__main__':
    data = np.load('data/scale_data.npy')
    value = np.load('data/value.npy')

    data = torch.from_numpy(data)

    # 加载模型
    network = Ann()
    checkpoint = torch.load('./model/model_992.pt')
    network.load_state_dict(checkpoint)

    out = network(data)
    out = out.detach().numpy()
    print(out)
    out = np.exp(out)
    out = out.reshape(-1)

    mse = round(np.sqrt(mean_squared_error(value, out)), 6)
    print('mse:{:.6f}'.format(mse))
    out = pd.DataFrame(out)
    value = pd.DataFrame(value)
    temp = pd.concat([out, value], axis=1)
    print(temp.corr("pearson"))

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(value)), value)
    plt.scatter(range(len(out)), out)
    plt.show()

    # print(out,mse)
    # res = pd.DataFrame(columns=['true', 'pred'])
    # res['true'] = value
    # res['pred'] = out
    #
    # writer = pd.ExcelWriter('true_pred.xlsx')
    # res.to_excel(writer,index=False)
    # writer.save()

    # res = pd.read_excel('dmax_true_pred.xlsx')
    #
    # # 绘制数据和线性回归模型拟合图
    # g = sns.regplot('true', "pred", res)
    #
    # # 相关系数
    # pearson = round(res['true'].corr(res['pred']), 2)
    #
    # # 均方误差
    # mse = round(np.sqrt(mean_squared_error(res['true'], res['pred'])), 3)
    #
    # plt.text(0.3, 0.8, 'rmse: {} \npearson: {}'.format(mse, pearson), size=14, transform=g.transAxes)
    #
    # plt.show()
