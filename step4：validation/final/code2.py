import pandas as pd

if __name__ == '__main__':
    data = pd.read_excel('../data/res_data.xlsx')
    idx = []
    for i in range(len(data)):
        if data.loc[i, 'Tg'] > data.loc[i, 'Tx'] or data.loc[i, 'Tx'] > data.loc[i, 'Tl'] or data.loc[i, 'Tg'] > \
                data.loc[i, 'Tl']:
           idx.append(i)

    data = data.drop(idx)

    writer = pd.ExcelWriter('../data/res_data_1.xlsx')
    data.to_excel(writer,index=False)
    writer.save()

