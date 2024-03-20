
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import CustomDataSet
import time
import math

from models.LA_WKN_BiGRU import LA_WKN_BiGRU
from utils import *

work_condition = 1
batch_size = 32

def Test_pipeline(trained_pth, batch_size, sX, work_condition):
    if work_condition == 1:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_4',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_5',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_6',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_7']
    if work_condition == 2:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_4',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_5',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_6',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_7']

    for test_target in Test_set:
        Bearing_name = test_target.split('/')[-1]
        test_data = CustomDataSet(test_target, work_condition, mode='test')
        test_loader = DataLoader(test_data, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        sX = SSO_hp_trans(sX)
        model = LA_WKN_BiGRU(sX).to(device)

        model.load_state_dict(torch.load(trained_pth))
        model.eval()

        print("target: {}".format(test_target))
        ans = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                # print("size: {}".format(outputs.size()))
                # print(outputs)
                vals = outputs.tolist()
                for val in vals:
                    ans.append(val)
        # print(ans)
                    
        output_2_csv(Bearing_name, ans)
        output_2_plot(Bearing_name, ans)

    return Bearing_name, ans

# test

if __name__ == "__main__":
    # setup
    exp_topic = 'noSSO'
    exp_num = 5
    batch_size = 32
    work_condition = [1,2]
    X = [0.001, 32, 64, 16, 32, 32, 3, 1, 0.5, 64, 0.3, 0.5, 0.5]

    start_time2 = time.time()
    for wc in work_condition:
        trained_pth = 'F:/git_repo/WKN_SSO/result/pth/'+exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st.pth'
        tmp_result = []
        bearing_name, tmp_result = Test_pipeline(trained_pth, batch_size, X, wc)
        output_2_csv(bearing_name, tmp_result)
        output_2_plot(bearing_name, tmp_result)
    end_time2 = time.time()
    test_time = end_time2-start_time2

    print("Test Finish !")
    print("Test Time = {}".format(test_time))