import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from capla import CAPLA, test

if __name__ == '__main__':
    print(sys.argv)
    argvs = sys.argv

    SHOW_PROCESS_BAR = True
    data_path = '../data/'
    td_data_path = "../3Ddata"
    path = Path(f'../saveModel/CAPLA_bestModel')
    device = torch.device("cuda:0")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    max_seq_len = 1000
    max_pkt_len = 64
    max_smi_len = 150

    batch_size = 290
    interrupt = None


    model = CAPLA()
    model = model.to(device)

    data_loaders = {phase_name: DataLoader(MyDataset(data_path, td_data_path, phase_name,
        max_seq_len, max_pkt_len, max_smi_len, pkt_window=None,pkt_stride=None),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        shuffle=False)
        for phase_name in[argvs[1]]
        }


    loss_function = nn.MSELoss(reduction='sum')

    model.load_state_dict(torch.load(path / 'best_model.pt',map_location='cuda:0'))

    for _p in [argvs[1]]:
       performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
       print(f'{_p}:')
       for k, v in performance.items():
           print(f'{k}: {v}\n')
       print()

