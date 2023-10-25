
import numpy as np
import pandas as pd
import os

def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = -1 + 2 * (data - min_val) / (max_val - min_val)

    return scaled_data

def get_health_index(root_dir, file_path):
    bearing_name = os.path.join(root_dir, file_path.split('/')[-2])
    file_num = int(file_path.split('/')[-1].split('_')[-1].split('.')[0])

    folder_tot = 0
    for filename in os.listdir(bearing_name):
        if filename.endswith('.csv'):
            folder_tot += 1
        
    hi = np.linspace(1,0,folder_tot)

    return hi[file_num-1]