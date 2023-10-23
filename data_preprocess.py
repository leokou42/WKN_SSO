
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# folder_name = './viberation_dataset/Learning_set/Bearing1_1/'

class Data():

    def __init__(self):
        self.data = self.get_data()
        self.hrz_data = self.get_hrz_data()
        self.label = self.get_label()

    def get_data(self, folder_name, mode = 'step'):
        files = [file for file in sorted(os.listdir(folder_name)) if 'acc' in file]

        data_list = []
        for file in files :
            df = pd.read_csv(f'{folder_name}/{file}', header=None)
            df = df.drop(5, axis=1)
            if mode == 'step' :
                df = df.iloc[::100, :]
            elif mode == 'max' :
                index = df.iloc[:, -1].argmax()
                df = df.iloc[index, :].to_frame().T
            elif mode == 'all':
                pass

            data_list += df.values.tolist()
            
        data = np.array(data_list)
        return data

    def get_hrz_data(self, folder_name):
        files = [file for file in sorted(os.listdir(folder_name)) if 'acc' in file]
        
        hrz_data_list = []
        for file in files:
            df_hrz = pd.read_csv(f'{folder_name}/{file}', header=None)
            df_hrz = df_hrz[4]
            hrz_data_list.append(df_hrz) 
        hrz_data = np.array(hrz_data_list)

        # data normalize, minmax, [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_data = scaler.fit_transform(hrz_data)

        return normalized_data
    
    def get_label(self, folder_name, plot = 'False'):
        files = [file for file in sorted(os.listdir(folder_name)) if 'acc' in file]
        df_points = 0
        for file in files:
            df = pd.read_csv(f'{folder_name}/{file}', header=None)
            df_points += len(df)

        label_data = np.linspace(1, 0, df_points)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(label_data)
            plt.xlabel('number of data')
            plt.ylabel('Health index')
            plt.title('raw signal data')
            plt.grid(True)
            plt.show()


