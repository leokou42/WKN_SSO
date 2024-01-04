
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

def Test_pipeline(trained_pth, Test_set, batch_size, work_condition):
    Bearing_name = Test_set.split('/')[-1]
    test_data = CustomDataSet(Test_set, work_condition, mode='test')
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = LA_WKN_BiGRU()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = LA_WKN_BiGRU().to(device)

    model.load_state_dict(torch.load(trained_pth))
    model.eval()

    print("target: {}".format(Test_set))
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

    return Bearing_name, ans

# test
exp_topic = 'SSO'
exp_num = 4

start_time2 = time.time()
for wc in work_condition:
    trained_pth = 'F:/git_repo/WKN_SSO/result/pth/'+exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st.pth'
    if wc == 1:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_3',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_4',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_5',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_6',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_7']
    elif wc == 2:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_3',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_4',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_5',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_6',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_7']
    elif wc == 3:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing3_3']
        
    for test_set in Test_set:
        tmp_result = []
        bearing_name, tmp_result = Test_pipeline(trained_pth, test_set, hyper_parameter[0], wc)
        output_2_csv(bearing_name, tmp_result)
        output_2_plot(bearing_name, tmp_result)
end_time2 = time.time()
test_time = end_time2-start_time2

print("Test Finish !")
print("Test Time = {}".format(test_time))