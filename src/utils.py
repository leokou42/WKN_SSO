
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    # scaler[0~1]
    # scaled_data = (data - min_val) / (max_val - min_val) 
    
    # scaler[-1~1]
    scaled_data = -1 + 2 * (data - min_val) / (max_val - min_val)

    return scaled_data

def get_health_index(root_dir, file_path):
    # bearing_name = os.path.join(root_dir, file_path.split('/')[-2])
    # file_num = int(file_path.split('/')[-1].split('_')[-1].split('.')[0])
    bearing_name = os.path.join(root_dir, file_path.split('\\')[-2])
    file_num = int(file_path.split('\\')[-1].split('_')[-1].split('.')[0])

    folder_tot = 0
    for filename in os.listdir(bearing_name):
        if filename.endswith('.csv'):
            folder_tot += 1
        
    hi = np.linspace(1,0,folder_tot)

    return hi[file_num-1]

def check_full_data(file_dir):
    df = pd.read_csv(file_dir, header=None)
    row_count = len(df)
    if row_count == 2560:
        return True
    else:
        return False

def output_2_csv(file_name, result):
    dict = {'health index' : result}
    df = pd.DataFrame(dict)
    file_name = 'F:/git_repo/WKN_SSO/result/' + file_name + '.csv'
    df.to_csv(file_name)
    print("{} saved".format(file_name))

def output_2_plot(file_name, result, show_pic = False):
    plt.figure(figsize=(10, 6))

    plt.plot(result)
    plt.title(file_name)
    plt.xlabel('Time')
    plt.ylabel('Health Index')
    pic_name = 'F:/git_repo/WKN_SSO/result/' + file_name + '.png'
    plt.savefig(pic_name)

    if show_pic == True:
        plt.show()

    print("{} picture saved".format(file_name)) 

