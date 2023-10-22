
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


folder_name = './viberation_dataset/Learning_set/Bearing1_1/'
files = [file for file in sorted(os.listdir(folder_name)) if 'acc' in file]
# print(len(files))

hrz_data_list = []
for file in files:
    df_hrz = pd.read_csv(f'{folder_name}/{file}', header=None)
    df_hrz = df_hrz[4]
    hrz_data_list.append(df_hrz)
    
hrz_data = np.array(hrz_data_list)
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_data = scaler.fit_transform(hrz_data)
