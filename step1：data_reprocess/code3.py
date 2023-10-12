'''
将不同数据表的数据汇总在一起
'''
import pandas as pd
import numpy as np
from copy import deepcopy

# Agasdawd
if __name__ == '__main__':
    data = pd.read_excel('../data/data.xlsx')
    print(len(data))

    paths = ['30','442','665','695','Dmax Dataset','GFA Dataset']
    for path in paths:
        data1 = pd.read_excel('../data/{}.xlsx'.format(path))
        data = data.append(data1)

    print(len(data))

    writer = pd.ExcelWriter('../data/data1.xlsx')
    data.to_excel(writer,index=False)
    writer.save()

