import pandas as pd
import numpy as np
import torch
from gfa_model import Ann

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    data = pd.read_excel('../data/data.xlsx')

    scale_data = np.load('../data/gfa_data.npy')
    scale_data = torch.from_numpy(scale_data)

    # 预测GFA类别
    gfa_network = Ann(in_dim=25, out_dim=3)
    params = torch.load('gfa.pt')
    gfa_network.load_state_dict(params)

    gfa_outputs = gfa_network(scale_data).detach().numpy().argmax(1)
    data['GFA'] = gfa_outputs

    print(len(gfa_outputs))
    print(sum(gfa_outputs == 0))
    print(sum(gfa_outputs == 1))
    print(sum(gfa_outputs == 2))

    # writer = pd.ExcelWriter('../data/data.xlsx')
    # data.to_excel(writer, index=False)
    # writer.save()
