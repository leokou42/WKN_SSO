
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import csv
import torch

def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    # scaler[0~1]
    # scaled_data = (data - min_val) / (max_val - min_val) 
    # scaler[-1~1]
    scaled_data = -1 + 2 * (data - min_val) / (max_val - min_val)

    return scaled_data

def two_stage_hi(twist_point, slope, l, draw=False):
    hi1_length = int(l * slope)
    hi2_length = l - hi1_length  # 確保總長度等於 l
    hi1 = np.linspace(1, twist_point, hi1_length)
    hi2 = np.linspace(twist_point, 0, hi2_length)
    hi = np.concatenate([hi1, hi2])

    if draw == 1:
        plt.plot(hi)
        plt.xlabel('Time')
        plt.ylabel('Health Index')
        plt.title('Plot of HI')
        plt.show

    return hi

def folder_total_len(root_dir, file_path):
    bearing_name = os.path.join(root_dir, file_path.split('\\')[-2])
    # bearing_name = os.path.join(root_dir, file_path.split('/')[-2])

    folder_total = 0
    for filename in os.listdir(bearing_name):
        if filename.endswith('.csv'):
            folder_total += 1
    
    return folder_total

def get_health_index(root_dir, file_path, hi_type=1, two_stage_hp=[0.6, 0.6]):
    folder_tot = folder_total_len(root_dir, file_path)
    file_num = int(file_path.split('/')[-1].split('_')[-1].split('.')[0])

    if hi_type == 1:
        hi = np.linspace(1,0,folder_tot)
    elif hi_type == 2:
        hi = two_stage_hi(two_stage_hp[0],two_stage_hp[1], folder_tot)

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
    file_name = 'F:/git_repo/WKN_SSO/result/csv/' + file_name + '.csv'
    df.to_csv(file_name)
    print("{} saved".format(file_name))

def output_2_plot(file_name, result, show_pic = False):
    plt.figure(figsize=(10, 6))

    MA_result = moving_avg(result, 10)
    plt.plot(result, label='prediction', color='blue')
    plt.plot(MA_result, label='MA prediction', color='red')

    plt.title(file_name)
    plt.xlabel('Time')
    plt.ylabel('Health Index')
    plt.legend()

    pic_name = 'F:/git_repo/WKN_SSO/result/plots/' + file_name + '.png'
    # pic_name = file_name + '.png'
    plt.savefig(pic_name)

    if show_pic == True:
        plt.show()

    print("{} picture saved".format(file_name)) 

def loss_2_plot(file_name, loss, mse, show_pic = False):
    plt.figure(figsize=(10, 6))

    plt.plot(loss, label='loss', color='blue')
    plt.plot(mse, label='MSE', color='red')

    plt.title(file_name)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()

    pic_name = 'F:/git_repo/WKN_SSO/result/train_loss/' + file_name + '.png'
    plt.savefig(pic_name)

    if show_pic == True:
        plt.show()

    print("{} picture saved".format(file_name))

# 生成隨機解組合
def generate_random_numbers(number_range):
    random_numbers = []
    for min_value, max_value in number_range:
        random_value = random.uniform(min_value, max_value)
        if random_value > 1:
          random_value = int(random_value)
        random_numbers.append(random_value)

    return random_numbers

# 生成隨機數
def random_select(n, random_number_range):
    return random.uniform(random_number_range[n][0],random_number_range[n][1])

def find_min_key_value(dictionary):
    if not dictionary:
        return None, None  # 如果字典是空的，返回None

    min_key = min(dictionary.keys())
    min_value = dictionary[min_key]
    
    return min_key, min_value

def train_2_csv(csv_file, list1, list2):
    csv_file = csv_file + '.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['List 1', 'List 2'])
        for item1, item2 in zip(list1, list2):
            writer.writerow([item1, item2])

    print(f'資料已寫入到 {csv_file} 檔案中。')

def SSO_2_csv(filename, value_to_x_dict):
    header = ['score', 'params', 'indices']
    # Writing to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(header)
        
        # Write data
        for score, (params, indices) in value_to_x_dict.items():
            writer.writerow([score, params, indices])
    
    print(filename)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def SSO_hp_trans(iX):
    iX[0] = iX[0]/100000
    iX[8] = iX[8]/100
    iX[10] = iX[10]/100
    iX[11] = iX[11]/100
    iX[12] = iX[12]/100

    return iX