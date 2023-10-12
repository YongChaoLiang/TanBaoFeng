'''
查看数据信息
'''

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_excel('../data/res_data.xlsx')
    print('数据总量：', len(data))
    # 查看 GFA 三种种类的数目
    bmg = data[data['GFA'] == 'BMG']
    ribbon = data[data['GFA'] == 'RMG']
    ct = data[data['GFA'] == 'CRA']
    print('RMG数据量：', len(ribbon))
    print('CRA数据量：', len(ct))
    print('BMG数据量：', len(bmg))
    # 查看 BMG Dmax 的数目
    Dmax = data[~data['Dmax'].isnull()]
    Dmax = Dmax[Dmax['Dmax'] > 0.15]
    print('Dmax数据量：', len(Dmax))
    # 查看 Tg 数目
    Tg = data[~data['Tg'].isnull()]
    print('Tg数据量：', len(Tg))
    # 查看 Tx 数目
    Tx = data[~data['Tx'].isnull()]
    print('Tx数据量：', len(Tx))
    # 查看 Tl 数目
    Tl = data[~data['Tl'].isnull()]
    print('Tl数据量：', len(Tl))

    # Dmax 数据分布
    print(np.asarray(Dmax['Dmax']))
    dmax = np.log(np.asarray(Dmax['Dmax'],dtype=float))

    fig, ax = plt.subplots()

    plt.hist(Dmax['Dmax'], bins=20, rwidth=0.8)

    # plt.title('Dmax 分布情况')
    plt.xlabel('大小(mm)')
    plt.ylabel('数量')
    fig.patch.set_alpha(0.)
    plt.show()
