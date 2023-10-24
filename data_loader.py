
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# folder_name = './viberation_dataset/Learning_set/Bearing1_1/'

class Data():

    def __init__(self, folder_name):
        self.folder_name = folder_name

    def get_hrz_data(self):
        files = [file for file in sorted(os.listdir(self.folder_name)) if 'acc' in file]
        
        hrz_data_list = []
        for file in files:
            df_hrz = pd.read_csv(f'{self.folder_name}/{file}', header=None)
            df_hrz = df_hrz[4]
            hrz_data_list.append(df_hrz) 
            
            # hrz_data_list += df_hrz.values.tolist()

        hrz_data = np.array(hrz_data_list)

        # data normalize, minmax, [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        hrz_data = scaler.fit_transform(hrz_data)

        hrz_data = hrz_data.reshape(-1)

        return hrz_data
    
    def get_label(self):
        files = [file for file in sorted(os.listdir(self.folder_name)) if 'acc' in file]
        
        df_points = 0
        for file in files:
            df = pd.read_csv(f'{self.folder_name}/{file}', header=None)
            df_points += len(df)
        
        label_data = np.linspace(1, 0, df_points)

        return label_data

