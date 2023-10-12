"""
对其它指标进行计算
"""

import pandas as pd

if __name__ == '__main__':
    data = pd.read_excel('../data/gp_data.xlsx')
    res = pd.DataFrame(columns=['Dmax', 'Trg', 'γ', 'ΔTrg', 'β1', 'ϕ', 'β2', 'ω', 'γc', 'β', 'Gp', 'χ'])

    Tg = data['Tg']
    Tx = data['Tx']
    Tl = data['Tl']

    res['Dmax'] = data['Dmax']
    res['Trg'] = Tg / Tl
    res['γ'] = Tx / (Tg + Tl)
    res['ΔTrg'] = (Tx - Tg) / (Tl - Tg)
    res['β1'] = Tx / Tg + Tg / Tl
    res['ϕ'] = Tg / Tl * ((Tx - Tg) / Tg) ** 0.143
    res['β2'] = Tx * Tg / (Tl - Tx) ** 2
    res['ω'] = Tl * (Tl + Tx) / (Tx * (Tl - Tx))
    res['γc'] = (3 * Tx - 2 * Tg) / Tl
    res['β'] = Tg / Tx - Tg / (1.3 * Tl)
    res['Gp'] = Tg * (Tx - Tg) / (Tl - Tx) ** 2
    res['χ'] = (Tx - Tg) / (Tl - Tx) * (Tx / (Tl - Tx)) ** 1.47

    writer = pd.ExcelWriter('../data/reported_data.xlsx')
    res.to_excel(writer, index=False)
    writer.save()
