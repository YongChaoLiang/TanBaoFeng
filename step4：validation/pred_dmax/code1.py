'''
处理数据，计算25个特征，并进行缩放处理
'''
import numpy as np
import pandas as pd
import re
from sklearn import preprocessing

np.set_printoptions(suppress=True)


# f: 元素占比   x：元素属性值
def delta_x(f, x):
    m = x.mean()
    s = 0
    for each1, each2 in zip(f, x):
        s += each1 * (1 - each2 / m) ** 2
    return np.sqrt(s)


def overline_x(f, x):
    s = 0
    for each1, each2 in zip(f, x):
        s += each1 * each2
    return s


def va(f, x):
    s = 0
    for each1, each2 in zip(f, x):
        s += each1 * 4 / 3 * np.pi * each2 ** 3
    return s


def smix(f):
    s = 0
    for each in f:
        if each > 0:
            s += each * np.log(each)
    return -8.314 * s


def hmix(h, f, eles):
    s = 0
    for i, ai in enumerate(f):
        for j, aj in enumerate(f):
            s += h.loc[eles[i], eles[j]] * ai * aj
    return 4 * s


def preprocess(h, info, compositions):
    res = np.zeros((len(compositions), 25))
    pros = ['Eea(ev)', 'I1(ev)', 'I2(ev)', 'Tm(K)', 'Hf(kJ/mol)', 'VEC', 'XP', 'XM', 'Cp(J/molK)', 'K(W/m)/K300K',
            'Rm(nm)']
    temp = None

    for idx, composition in enumerate(compositions):
        # 分离出元素及占比
        ele = re.findall('[a-zA-Z]+', composition)
        dig = np.asarray(list(map(float, re.findall('[0-9]+(?:[.]{1}[0-9]+){0,1}', composition))))
        dig = dig / 100
        for i in range(11):
            temp = info[info['Element'].map(lambda x: x in ele)][pros[i]]
            temp = np.asarray(temp)

            res[idx][2 * i] = delta_x(dig, temp)
            res[idx][2 * i + 1] = overline_x(dig, temp)

        res[idx][22] = va(dig, temp)
        res[idx][23] = smix(dig)
        try:
            res[idx][24] = hmix(h, dig, ele)
        except:
            print(composition)
        # 打印进度
        if idx % 100 == 0:
            print('已完成：{}%'.format(idx / len(compositions) * 100))

    print(res.mean(axis=0))
    print(res.std(axis=0))

    res = preprocessing.scale(res)

    return res


if __name__ == '__main__':
    info = pd.read_excel('../../data/info.xlsx')
    data = pd.read_excel('../data/data.xlsx')
    h = pd.read_excel('../../data/1.xlsx', header=0, index_col=0)

    # 找出GFA下标
    idx = data['GFA'] == 2
    idx = np.argwhere(np.asarray(idx) == True).flatten()

    compositions = np.asarray(data['composition_1'])[idx]
    res = preprocess(h, info, compositions)
    np.save('../data/dmax_data.npy', res)

