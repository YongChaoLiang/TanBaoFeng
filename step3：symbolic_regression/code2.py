"""
进行符号回归分析
"""
import pickle
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer


def main():
    # 加载数据
    gp_data = pd.read_excel('./data/gp_data.xlsx')

    function_set = ['add', 'sub', 'mul', 'div']

    gp = SymbolicTransformer(generations=30, population_size=10000,
                             hall_of_fame=100, n_components=1,
                             function_set=function_set,
                             parsimony_coefficient=0.005,
                             verbose=1, n_jobs=3)

    gp.fit(gp_data[['(Tx−Tg)/(Tl−Tx)', '(Tx−Tg)/(Tl−Tg)', 'Tx/(Tl+Tg)', 'Tx/(Tl+Tx)', '(Tx+Tg)/(Tl−Tx)', 'Tg/(Tl−Tx)',
                    'Tx/Tl', 'Tx/(Tl−Tx)']],
           gp_data['Dmax'])

    with open('model/model8/gp_model.pkl', 'wb') as f:
        pickle.dump(gp, f)


if __name__ == '__main__':
    main()
