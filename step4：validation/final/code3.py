"""
计算数据
"""

import pandas as pd

if __name__ == '__main__':
    data = pd.read_excel('../data/res_data_2.xlsx')

    Tg = data['Tg']
    Tx = data['Tx']
    Tl = data['Tl']

    data['Tx/Tl'] = Tx / Tl
    data['Tx/Tg'] = Tx / Tg
    data['Tg/Tl'] = Tg / Tl
    data['Tx/(Tx-Tg)'] = Tx / (Tx - Tg)
    data['Tx/(Tx+Tg)'] = Tx / (Tx + Tg)
    data['Tx/(Tl−Tg)'] = Tx / (Tl - Tg)
    data['Tx/(Tl+Tg)'] = Tx / (Tl + Tg)
    data['Tx/(Tl−Tx)'] = Tx / (Tl - Tx)
    data['Tx/(Tl+Tx)'] = Tx / (Tl + Tx)
    data['Tg/(Tl+Tx)'] = Tg / (Tl + Tx)
    data['Tg/(Tl+Tg)'] = Tg / (Tl + Tg)
    data['Tg/(Tl−Tg)'] = Tg / (Tl - Tg)
    data['Tg/(Tl−Tx)'] = Tg / (Tl - Tx)
    # data['Tg/(Tx−Tg)'] = Tg / (Tx - Tg)
    # data['Tg/(Tx+Tg)'] = Tg / (Tx + Tg)
    data['Tl/(Tx−Tg)'] = Tl / (Tx - Tg)
    data['Tl/(Tx+Tg)'] = Tl / (Tx + Tg)
    # data['Tl/(Tl−Tg)'] = Tl / (Tl - Tg)
    # data['Tl/(Tl+Tg)'] = Tl / (Tl + Tg)
    # data['Tl/(Tl−Tx)'] = Tl / (Tl - Tx)
    # data['Tl/(Tl+Tx)'] = Tl / (Tl + Tx)
    data['(Tx+Tg)/(Tl+Tx)'] = (Tx + Tg) / (Tl + Tx)
    data['(Tx+Tg)/(Tl−Tg)'] = (Tx + Tg) / (Tl - Tg)
    data['(Tx−Tg)/(Tl+Tg)'] = (Tx - Tg) / (Tl + Tg)
    data['(Tx−Tg)/(Tl+Tx)'] = (Tx - Tg) / (Tl + Tx)
    data['(Tx+Tg)/(Tl+Tg)'] = (Tx + Tg) / (Tl + Tg)
    data['(Tx+Tg)/(Tl−Tx)'] = (Tx + Tg) / (Tl - Tx)
    data['(Tx−Tg)/(Tl−Tg)'] = (Tx - Tg) / (Tl - Tg)
    data['(Tx−Tg)/(Tl−Tx)'] = (Tx - Tg) / (Tl - Tx)
    # data['(Tx−Tg)/(Tx+Tg)'] = (Tx - Tg) / (Tx + Tg)
    # data['(Tl−Tg)/(Tl+Tg)'] = (Tl - Tg) / (Tl + Tg)
    # data['(Tl−Tg)/(Tl−Tx)'] = (Tl - Tg) / (Tl - Tx)
    # data['(Tl−Tg)/(Tl+Tx)'] = (Tl - Tg) / (Tl + Tx)
    # data['(Tl+Tg)/(Tl−Tx)'] = (Tl + Tg) / (Tl - Tx)
    # data['(Tl+Tg)/(Tl+Tx)'] = (Tl + Tg) / (Tl + Tx)
    # data['(Tl−Tx)/(Tl+Tx)'] = (Tl - Tx) / (Tl + Tx)

    # 保存excel数据
    writer = pd.ExcelWriter('../data/gp_data.xlsx')
    data.to_excel(writer, index=False)
    writer.save()

