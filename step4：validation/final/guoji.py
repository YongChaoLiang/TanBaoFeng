import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_excel('../data/reported_data.xlsx')
    data1 = pd.read_excel('../data/gp_data.xlsx')
    x0 = np.asarray(data1['(Tx−Tg)/(Tl−Tx)'])
    x1 = np.asarray(data1['(Tx−Tg)/(Tl−Tg)'])
    x2 = np.asarray(data1['Tx/(Tl+Tg)'])
    x3 = np.asarray(data1['Tx/(Tl+Tx)'])
    x4 = np.asarray(data1['(Tx+Tg)/(Tl−Tx)'])
    x5 = np.asarray(data1['Tg/(Tl−Tx)'])
    x6 = np.asarray(data1['Tx/Tl'])
    x7 = np.asarray(data1['Tx/(Tl−Tx)'])
    data['this work'] = (x3 - x1) * (x3 - x1) * x5 * x0 * x0

    y_range = (0, 15)

    # plt.style.use('seaborn-bright')

    plt.subplot(4, 3, 1)
    g = sns.regplot('χ', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['χ'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\chi$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 2)
    g = sns.regplot('Gp', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['Gp'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$G_p$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 3)
    g = sns.regplot('Trg', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['Trg'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$T_{rg}$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 4)
    g = sns.regplot('β2', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['β2'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\beta_2$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 5)
    g = sns.regplot('ΔTrg', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['ΔTrg'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\Delta T_{rg}$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 6)
    g = sns.regplot('ω', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['ω'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\omega$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 7)
    g = sns.regplot('γc', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['γc'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\Upsilon_c$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 8)
    g = sns.regplot('β1', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['β1'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\beta_1$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 9)
    g = sns.regplot('γ', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['γ'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\gamma$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 10)
    g = sns.regplot('β', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['β'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel('特征$\\beta$\'的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(-pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 11)
    g = sns.regplot('ϕ', "Dmax", data, scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['ϕ'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\Phi$的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.subplot(4, 3, 12)
    g = sns.regplot('this work', "Dmax", data, color='orange', scatter_kws={'s': 20, 'alpha': 0.7})
    pearson = round(data['this work'].corr(data['Dmax']), 4)
    plt.ylabel(r'$D_{max}$/mm')
    plt.xlabel(r'特征$\alpha$(本次工作)的值')
    plt.ylim(y_range)
    plt.text(0.43, 0.90, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

    plt.show()
