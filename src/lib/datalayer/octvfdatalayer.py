import os
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

class OctVfDataset(Dataset):
    """OctVfDataset"""

    def __init__(self, data_dir, text_file):
        self.vf_names = []
        self.oct_names = []
        self.labels = []
        
        # load annotations
        with open(text_file, 'r') as fin:
            for line in fin:
                line = line.strip()
                vf_name = line.split('.mat')[0] + '.mat'
                oct_label = line.split('.mat ')[1]
                oct_name = oct_label[:-2]
                label = int(oct_label[-1])

                self.vf_names.append(os.path.join(data_dir, vf_name))
                self.oct_names.append(os.path.join(data_dir, oct_name))
                self.labels.append(label)

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, idx):
        # load OCT data
        oct_data = np.load(self.oct_names[idx])
        # load VF-PDP data
        vf_data = sio.loadmat(self.vf_names[idx])
        
        # convert objects to Tensor
        vf_data = torch.from_numpy(vf_data)
        oct_data = torch.from_numpy(oct_data)
        label = self.labels[idx]
        sample = {'oct_data': oct_data, 'vf_data': vf_data, 'label': label}

        return sample
