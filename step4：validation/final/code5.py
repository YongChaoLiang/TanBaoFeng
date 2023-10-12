import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
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

    writer = pd.ExcelWriter('../data/wdads.xlsx')
    data.to_excel(writer,index=False)
    writer.save()

    # y_range = (0, 15)
    #
    # # plt.style.use('seaborn-bright')
    # plt.figure(figsize=(15, 20))
    # # plt.figure(dpi=100)
    # matplotlib.rcParams.update({'font.size': 20})
    # text_fontsize = 20
    # scatter_kws = {'s': 20, 'alpha': 0.7}
    # line_kws = {"color": "red"}
    #
    # plt.subplot(4, 3, 1)
    # g = sns.regplot('χ', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['χ'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\chi$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 2)
    # g = sns.regplot('Gp', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['Gp'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $G_p$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 3)
    # g = sns.regplot('Trg', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['Trg'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $T_{rg}$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 4)
    # g = sns.regplot('β2', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['β2'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\beta_2$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 5)
    # g = sns.regplot('ΔTrg', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['ΔTrg'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\Delta T_{rg}$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 6)
    # g = sns.regplot('ω', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['ω'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\omega$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 7)
    # g = sns.regplot('γc', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['γc'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\Upsilon_c$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 8)
    # g = sns.regplot('β1', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['β1'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\beta_1$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 9)
    # g = sns.regplot('γ', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['γ'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\gamma$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 10)
    # g = sns.regplot('β', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['β'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel('The criterion $\\beta$\' value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(-pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 11)
    # g = sns.regplot('ϕ', "Dmax", data, scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['ϕ'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\Phi$ value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.subplot(4, 3, 12)
    # g = sns.regplot('this work', "Dmax", data, color='orange', scatter_kws=scatter_kws, line_kws=line_kws)
    # pearson = round(data['this work'].corr(data['Dmax']), 4)
    # plt.ylabel(r'$D_{max}$/mm')
    # plt.xlabel(r'The criterion $\theta$ (this work) value')
    # plt.ylim(y_range)
    # plt.text(0.43, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
    #          fontdict={'family': 'Times New Roman', 'size': text_fontsize, 'weight': 'bold'})
    #
    # plt.tight_layout()
    # plt.show()
