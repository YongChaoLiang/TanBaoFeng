import pandas as pd
import numpy as np
from t_model import Ann
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    data = pd.read_excel('../data/data.xlsx')

    scale_data = np.load('../data/t_data.npy')
    scale_data = torch.from_numpy(scale_data)

    t_network = Ann(in_dim=25, out_dim=3)
    params = torch.load('t.pt')
    t_network.load_state_dict(params)

    t_outputs = t_network(scale_data).detach().numpy()
    t_outputs = np.exp(t_outputs)

    idx = data['GFA'] == 2
    idx = np.argwhere(np.asarray(idx) == True).flatten()

    title = ['Tg', 'Tx', 'Tl']
    for i in range(3):
        data[title[i]][idx] = t_outputs[:, i]

    writer = pd.ExcelWriter('../data/data.xlsx')
    data.to_excel(writer, index=False)
    writer.save()
