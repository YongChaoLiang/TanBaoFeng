"""
计算数据
"""

import pandas as pd

if __name__ == '__main__':
    data = pd.read_excel('../data/res_data.xlsx')
    df = pd.DataFrame(columns=list(data))

    for i in range(len(data)):
        if all(~data.iloc[i].isnull()):
            if not (data.loc[i, 'Tg'] == data.loc[i, 'Tl']
                    or data.loc[i, 'Tg'] == data.loc[i, 'Tx']
                    or data.loc[i, 'Tx'] == data.loc[i, 'Tl']):
                df = df.append(data.iloc[i], ignore_index=True)

    Tg = df['Tg']
    Tx = df['Tx']
    Tl = df['Tl']

    df['Tx/Tl'] = Tx / Tl
    df['Tx/Tg'] = Tx / Tg
    df['Tg/Tl'] = Tg / Tl
    df['Tx/(Tx-Tg)'] = Tx / (Tx - Tg)
    df['Tx/(Tx+Tg)'] = Tx / (Tx + Tg)
    df['Tx/(Tl−Tg)'] = Tx / (Tl - Tg)
    df['Tx/(Tl+Tg)'] = Tx / (Tl + Tg)
    df['Tx/(Tl−Tx)'] = Tx / (Tl - Tx)
    df['Tx/(Tl+Tx)'] = Tx / (Tl + Tx)
    df['Tg/(Tl+Tx)'] = Tg / (Tl + Tx)
    df['Tg/(Tl+Tg)'] = Tg / (Tl + Tg)
    df['Tg/(Tl−Tg)'] = Tg / (Tl - Tg)
    df['Tg/(Tl−Tx)'] = Tg / (Tl - Tx)
    # df['Tg/(Tx−Tg)'] = Tg / (Tx - Tg)
    # df['Tg/(Tx+Tg)'] = Tg / (Tx + Tg)
    df['Tl/(Tx−Tg)'] = Tl / (Tx - Tg)
    df['Tl/(Tx+Tg)'] = Tl / (Tx + Tg)
    # df['Tl/(Tl−Tg)'] = Tl / (Tl - Tg)
    # df['Tl/(Tl+Tg)'] = Tl / (Tl + Tg)
    # df['Tl/(Tl−Tx)'] = Tl / (Tl - Tx)
    # df['Tl/(Tl+Tx)'] = Tl / (Tl + Tx)
    df['(Tx+Tg)/(Tl+Tx)'] = (Tx + Tg) / (Tl + Tx)
    df['(Tx+Tg)/(Tl−Tg)'] = (Tx + Tg) / (Tl - Tg)
    df['(Tx−Tg)/(Tl+Tg)'] = (Tx - Tg) / (Tl + Tg)
    df['(Tx−Tg)/(Tl+Tx)'] = (Tx - Tg) / (Tl + Tx)
    df['(Tx+Tg)/(Tl+Tg)'] = (Tx + Tg) / (Tl + Tg)
    df['(Tx+Tg)/(Tl−Tx)'] = (Tx + Tg) / (Tl - Tx)
    df['(Tx−Tg)/(Tl−Tg)'] = (Tx - Tg) / (Tl - Tg)
    df['(Tx−Tg)/(Tl−Tx)'] = (Tx - Tg) / (Tl - Tx)
    # df['(Tx−Tg)/(Tx+Tg)'] = (Tx - Tg) / (Tx + Tg)
    # df['(Tl−Tg)/(Tl+Tg)'] = (Tl - Tg) / (Tl + Tg)
    # df['(Tl−Tg)/(Tl−Tx)'] = (Tl - Tg) / (Tl - Tx)
    # df['(Tl−Tg)/(Tl+Tx)'] = (Tl - Tg) / (Tl + Tx)
    # df['(Tl+Tg)/(Tl−Tx)'] = (Tl + Tg) / (Tl - Tx)
    # df['(Tl+Tg)/(Tl+Tx)'] = (Tl + Tg) / (Tl + Tx)
    # df['(Tl−Tx)/(Tl+Tx)'] = (Tl - Tx) / (Tl + Tx)

    # 保存excel数据
    writer = pd.ExcelWriter('./data/gp_data.xlsx')
    df.to_excel(writer, index=False)
    writer.save()

