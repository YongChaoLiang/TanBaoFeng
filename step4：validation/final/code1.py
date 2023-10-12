import pandas as pd

if __name__ == '__main__':
    data = pd.read_excel('../data/data.xlsx')
    temp = data[data['GFA']==2]

    writer = pd.ExcelWriter('../data/res_data.xlsx')
    temp.to_excel(writer,index=False)
    writer.save()