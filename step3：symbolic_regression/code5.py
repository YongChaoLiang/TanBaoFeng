import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import pandas as pd
import numpy as np
import math


# 计算相关度
def computeCorrelation(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0, len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar ** 2
        varY += difYYbar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST


# 计算R平方
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot
    return results


if __name__ == '__main__':
    # plt.figure(figsize=(24, 6),dpi=200)
    # matplotlib.rcParams.update({'font.size': 22})

    data = pd.read_excel('./data/reported_data.xlsx')
    data1 = pd.read_excel('./data/gp_data_1.xlsx')
    x0 = np.asarray(data1['(Tx−Tg)/(Tl−Tx)'])
    x1 = np.asarray(data1['(Tx−Tg)/(Tl−Tg)'])
    x2 = np.asarray(data1['Tx/(Tl+Tg)'])
    x3 = np.asarray(data1['Tx/(Tl+Tx)'])
    x4 = np.asarray(data1['(Tx+Tg)/(Tl−Tx)'])
    x5 = np.asarray(data1['Tg/(Tl−Tx)'])
    x6 = np.asarray(data1['Tx/Tl'])
    x7 = np.asarray(data1['Tx/(Tl−Tx)'])
    data['this work'] =  x5 * x0 * x0
    data['this work_1'] = (x1 - x3) * (x1 - x3) * x5 * x0 * x0

    fig = plt.figure(figsize=(8,4))

    plt.subplot(1, 2, 1)
    g = sns.regplot('this work', "Dmax", data, line_kws={"color": "red"})
    pearson = round(data['this work'].corr(data['Dmax']), 4)
    plt.text(0.33, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'})
    plt.xlabel(r'The criterion $T_g(T_x-T_g)^2/(T_l-T_x)^3$  value')
    plt.ylabel(r'$D_{max}$/mm')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    g = sns.regplot('this work_1', "Dmax", data, line_kws={"color": "red"})
    pearson = round(data['this work_1'].corr(data['Dmax']), 4)
    plt.text(0.33, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'})
    plt.xlabel(r'The criterion $\theta$ value')
    plt.ylabel(r'$D_{max}$/mm')
    plt.tight_layout()

    # plt.subplot(1, 3, 2)
    # g = sns.regplot('Gp', "Dmax", data, line_kws={"color": "red"})
    # pearson = round(data['Gp'].corr(data['Dmax']), 4)
    # plt.text(0.33, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': 32,'weight':'bold'})
    # # plt.xlabel(r'The criterion $G_p$ value (b)', fontdict={'family': 'Times New Roman', 'size': 32})
    # # plt.ylabel(r'$D_{max}$/mm', fontproperties='Times New Roman', size=32)
    # plt.tight_layout()
    #
    # plt.subplot(1, 3, 1)
    # g = sns.regplot('χ', "Dmax", data, line_kws={"color": "red"})
    # pearson = round(data['χ'].corr(data['Dmax']), 4)
    # plt.text(0.33, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': 32, 'weight': 'bold'})
    # # plt.xlabel(r'The criterion $χ$ value (a)',  fontdict={'family': 'Times New Roman', 'size': 32})
    # # plt.ylabel(r'$D_{max}$/mm', fontproperties='Times New Roman', size=32)
    # plt.tight_layout()
    #
    # plt.subplot(1, 3, 3)
    # g = sns.regplot('this work', "Dmax", data,color='orange', line_kws={"color": "red"})
    # pearson = round(data['this work'].corr(data['Dmax']), 4)
    # plt.text(0.33, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': 32, 'weight': 'bold'})
    # # plt.xlabel(r'The criterion $\theta$ value (c)',  fontdict={'family': 'Times New Roman', 'size': 32})
    # # plt.ylabel(r'$D_{max}$/mm', fontproperties='Times New Roman', size=32)
    # plt.tight_layout()

    plt.tight_layout()
    plt.show()
