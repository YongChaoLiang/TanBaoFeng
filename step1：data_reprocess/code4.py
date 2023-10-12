'''
对收集的数据全部进行去重处理，对于歧义数据，采用最大值
'''
import numpy as np
import pandas as pd
from collections import Counter


def mygfa(gfa):
    if 'BMG' in gfa:
        return 'BMG'
    elif 'RMG' in gfa:
        return 'RMG'
    else:
        return 'CRA'

if __name__ == '__main__':
    path = '../data/data1.xlsx'
    data = pd.read_excel(path)

    print(len(data))
    print(len(set(data['composition_1'])))

    comps = data['composition_1']

    counter = Counter(comps)
    for each in dict(counter):
        if counter[each] == 1:
            counter.pop(each)

    comps = list(counter)

    data1 = pd.DataFrame(columns=list(data))
    for each in comps:
        temp = data[data['composition_1'] == each]
        data = data.drop(data[data['composition_1'] == each].index)
        data1 = data1.append({
            'composition': list(temp['composition'])[0],
            'Tg': np.nanmax(temp['Tg']),
            'Tx': np.nanmax(temp['Tx']),
            'Tl': np.nanmax(temp['Tl']),
            'GFA': mygfa(list(temp['GFA'])),
            'Dmax': np.nanmax(temp['Dmax']),
            'composition_1': each
        }, ignore_index=True
        )
    print(len(data1))
    data = data.append(data1, ignore_index=True)

    print(len(data))

    writer = pd.ExcelWriter('../data/data2.xlsx')
    data.to_excel(writer, index=False)
    writer.save()
