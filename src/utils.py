
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
    
def moving_avg(data, window):
    MA_data = []
    for i in range(len(data)):
        tmp = 0
        if i < window:
            MA_data.append(data[i])
        else:
            start_point = i - window
            for j in range(start_point, i):
                tmp += data[j]
            
            MA_data.append(tmp / window)

    return MA_data

def output_2_csv(file_name, result):
    MA_result = moving_avg(result, 5)
    dict = {'health index' : result, 'MA_health index' : MA_result}
    df = pd.DataFrame(dict)
    file_name = 'F:/git_repo/WKN_SSO/result/' + file_name + '.csv'
    df.to_csv(file_name)
    print("{} saved".format(file_name))

def output_2_plot(file_name, result, show_pic = False):
    plt.figure(figsize=(10, 6))

    MA_result = moving_avg(result, 5)
    plt.plot(result, label='prediction', color='blue')
    plt.plot(MA_result, label='MA prediction', color='red')

    plt.title(file_name)
    plt.xlabel('Time')
    plt.ylabel('Health Index')
    plt.legend()

    # pic_name = 'F:/git_repo/WKN_SSO/result/' + file_name + '.png'
    pic_name = file_name + '.png'
    plt.savefig(pic_name)

    if show_pic == True:
        plt.show()

    print("{} picture saved".format(file_name)) 

