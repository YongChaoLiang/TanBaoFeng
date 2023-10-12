'''
对元素组成进行排序，方便收集数据时去重
'''

import re
from decimal import Decimal

import pandas as pd
import numpy as np

def ele_dig(s):
    index = [-1, -1, -1]
    for i in range(len(s)):
        if s[i] in ['[', '(', '{']:
            temp = s[:i]
            d = re.findall('[0-9]+(?:[.]{1}[0-9]+){0,1}', temp)
            if s[i] == '(':
                index[0] = len(d)
            elif s[i] == '[':
                index[1] = len(d)
            else:
                index[2] = len(d)

    dig = re.findall('[0-9]+(?:[.]{1}[0-9]+){0,1}', s)
    ele = re.findall('[a-zA-Z]+', s)
    dig = np.asarray(list(map(Decimal, dig)))

    i = 0
    while (i < 3) and (index[i] != -1):
        j = index[i]
        sum = Decimal('0')
        while (sum != 100) & (sum != 1):
            try:
                sum += dig[j]
            except:
                print(s)
                break
            j += 1
        for k in range(index[i], j):
            dig[k] /= sum
            try:
                dig[k] *= dig[j]
            except:
                print(s)
        dig = np.delete(dig, j)
        i += 1

    ele = np.asarray(ele)
    try:
        dig = dig[ele.argsort()]
    except:
        print(s)
    ele.sort()

    return ele, dig


if __name__ == '__main__':
    path = '../data/GFA Dataset.xlsx'
    data1 = pd.read_excel(path)

    for i in range(len(data1)):
        ele, dig = ele_dig(data1.loc[i, 'composition'])
        # 将数字转换为字符串
        temp = ''
        for each1, each2 in zip(ele, dig):
            temp = temp + each1 + str(each2)
        data1.loc[i, 'composition_1'] = temp

    writer = pd.ExcelWriter(path)
    data1.to_excel(writer, index=False)
    writer.save()


