import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    dmax = pd.read_excel('dmax_true_pred.xlsx')
    tg = pd.read_excel('Tg_true_pred.xlsx')
    tx = pd.read_excel('Tx_true_pred.xlsx')
    tl = pd.read_excel('Tl_true_pred.xlsx')

    plt.rc('font', family='Times New Roman', size='14')  # 设置字体样式、大小
    fontsize = 20
    fig = plt.figure(figsize=(8, 6), dpi=200)

    fig.add_subplot(2, 2, 1)
    g = sns.regplot('pred', 'true', dmax, line_kws={"color": "red"})
    pearson = round(dmax['true'].corr(dmax['pred']), 2)
    plt.xlabel('Measured $D_{max}$  (b)', fontsize=fontsize)
    plt.ylabel('Predicted $D_{max}$', fontsize=fontsize)
    # for i in range(len(dmax)):
    #     plt.text(dmax.loc[i, 'pred'], dmax.loc[i, 'true'], i + 1)
    plt.text(0.40, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'weight': 'bold'})
    plt.tight_layout()

    fig.add_subplot(2, 2, 2)
    g = sns.regplot('pred', 'true', tg, line_kws={"color": "red"})
    pearson = round(tg['true'].corr(tg['pred']), 2)
    plt.xlabel('Measured $T_g$  (c)', fontsize=fontsize)
    plt.ylabel('Predicted $T_g$', fontsize=fontsize)
    # for i in range(len(tg)):
    #     plt.text(tg.loc[i, 'pred'], tg.loc[i, 'true'], i + 1)
    plt.text(0.40, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'weight': 'bold'})
    plt.tight_layout()

    fig.add_subplot(2, 2, 3)
    g = sns.regplot('pred', 'true', tx, line_kws={"color": "red"})
    pearson = round(tx['true'].corr(tx['pred']), 2)
    plt.xlabel('Measured $T_x$  (d)', fontsize=fontsize)
    plt.ylabel('Predicted $T_x$', fontsize=fontsize)
    # for i in range(len(tx)):
    #     plt.text(tx.loc[i, 'pred'], tx.loc[i, 'true'], i + 1)
    plt.text(0.40, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'weight': 'bold'})
    plt.tight_layout()

    fig.add_subplot(2, 2, 4)
    g = sns.regplot('pred', 'true', tl, line_kws={"color": "red"})
    pearson = round(tl['true'].corr(tl['pred']), 2)
    plt.xlabel('Measured $T_l$  (e)', fontsize=fontsize)
    plt.ylabel('Predicted $T_l$', fontsize=fontsize)
    # for i in range(len(tl)):
    #     plt.text(tl.loc[i, 'pred'], tl.loc[i, 'true'], i + 1)
    plt.text(0.40, 0.80, r'$R$ = {}'.format(str(pearson)), transform=g.transAxes,
             fontdict={'weight': 'bold'})
    plt.tight_layout()

    plt.show()
