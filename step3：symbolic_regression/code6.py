"""
计算出36中特征中前5个与Dmax关系最大的特征
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gp_data = pd.read_excel('./data/gp_data_1.xlsx')

    data1 = gp_data[['Dmax'] + list(gp_data)[7:]]

    corrs = []
    for i in range(1, 24):
        corr = data1['Dmax'].corr(data1[list(data1)[i]])
        corrs.append(corr)

    corrs = np.asarray(corrs)
    corrs1 = np.abs(corrs)
    corrs1 = np.round(corrs1, 8)
    idx = np.argsort(corrs1)[::-1]
    print(idx)
    print(corrs1[idx])

    # plt.bar(x=[r'$\frac{(T_x−T_g)}{(T_l−T_x)}$',r'$\frac{(T_l−T_g)}{(T_l−T_x)}$',r'$\frac{(T_x−T_g)}{(T_l−T_g)}$',r'$\frac{T_l}{(T_l−T_x)}$',
    #            r'$\frac{T_x}{(T_l−T_x)}$']
    #         ,height=corrs1[idx],width=0.6)
    # plt.ylim(0.3,0.6)
    # plt.grid(alpha = 0.2)
    # plt.show()
