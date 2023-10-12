'''
生成数据 (Zr_i Co_j Al_k)95 X5
'''
import pandas as pd
import numpy as np

if __name__ == '__main__':

    gen_data = pd.DataFrame(columns=['composition_1', 'Dmax', 'Tg', 'Tx', 'Tl'])
    elements = ['W', 'Si', 'Ni']
    composition_1 = []
    count = 0
    for element in elements:
        for i in range(0, 101, 2):
            for j in range(0, 101 - i, 2):
                alloy = 'Zr' + str(95 * i / 100) + 'Co' + str(95 * j / 100) + \
                        'Al' + str(95 * (100 - i - j) / 100) + element + str(5)
                composition_1.append(alloy)
    gen_data['composition_1'] = composition_1

    writer = pd.ExcelWriter('./data/data.xlsx')
    gen_data.to_excel(writer,index=False)
    writer.save()