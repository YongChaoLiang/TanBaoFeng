import pandas as pd
import numpy as np
from dmax_model import Ann
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    data = pd.read_excel('../data/data.xlsx')

    scale_data = np.load('../data/dmax_data.npy')
    scale_data = torch.from_numpy(scale_data)

    # 加载最佳模型
    dmax_network = Ann()
    params = torch.load('dmax.pt')
    dmax_network.load_state_dict(params)

    dmax_outputs = dmax_network(scale_data).detach().numpy().reshape(-1)
    dmax_outputs = np.exp(dmax_outputs)

    idx = data['GFA'] == 2
    idx = np.argwhere(np.asarray(idx) == True).flatten()

    data['Dmax'][idx] = dmax_outputs

    writer = pd.ExcelWriter('../data/data.xlsx')
    data.to_excel(writer, index=False)
    writer.save()
