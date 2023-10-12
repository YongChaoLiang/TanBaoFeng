import pandas as pd
import numpy as np
# data = np.load('../data/data.npy')
# data = pd.DataFrame(data)
#
# print(data)
# writer = pd.ExcelWriter('../data/data2.xlsx')
# data.to_excel(writer, index=False)
# writer.save()

# data2 = np.load('../data/value.npy')
# data2 = pd.DataFrame(data2)
# # print(data2.shape)
# writer = pd.ExcelWriter('../data/dvalue2.xlsx')
# data2.to_excel(writer, index=False)
# writer.save()

# data2 = np.load('../data/log_value.npy')
# data2 = pd.DataFrame(data2)
# print(data2,data2.shape)

data3 = np.load('../data/scale_data.npy')
data3 = pd.DataFrame(data3)
writer = pd.ExcelWriter('../data/scala_data.xlsx')
data3.to_excel(writer, index=False)
writer.save()
print(data3)