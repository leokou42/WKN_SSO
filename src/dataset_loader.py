
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import *

class CustomDataSet(Dataset):
    def __init__(self, root_dir, work_condition,
                 transform=None, mode='train'):
        self.root_dir = root_dir
        self.work_condition = work_condition
        self.transform = transform
        self.mode = mode
        self.file_paths = self.get_file_paths()

    def get_file_paths(self):
        file_paths = []
        wk = 'Bearing'
        wk = wk + str(self.work_condition)

        for folder in os.listdir(self.root_dir):
            if wk in folder:
                folder_path = os.path.join(self.root_dir, folder)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.csv'):
                            pathes = os.path.join(folder_path, filename)
                            pathes = os.path.normpath(pathes)
                            file_paths.append(pathes)
        
        return file_paths

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        data = pd.read_csv(file_path, header=None, names=['hour', 'minute', 'second', 'microsecond', 'horiz accel', 'vert accel'])

        inputs = data['horiz accel'].values.astype(float)
        inputs = min_max_scale(inputs).reshape(1, -1)
        inputs = torch.from_numpy(inputs.astype(np.float32))
        # print(inputs.shape)

        if self.mode == 'train':
            label = get_health_index(self.root_dir, file_path)
            label = torch.tensor(label, dtype=torch.float32)

            if self.transform:
                inputs = self.transform(inputs)

            return inputs, label
        
        elif self.mode == 'test':
            if self.transform:
                inputs = self.transform(inputs)
            
            return inputs

        
    def __len__(self):
        return len(self.file_paths)
        
    
