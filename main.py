
import sys
sys.path.append("src")
import time 
import numpy as np
import random
import torch

from src.train import Train_pipeline
from src.test import Test_pipeline
from src.utils import *


# setup
hyper_parameter = [32, 50]   # [batch_size, num_epochs]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set'
train_vali = [Learning_set, Validation_set]
work_condition = [1,2]
exp_topic = 'noSSO'
exp_num = 4

#[learning rate, LA kernel, conv1 kernel, conv2 kernel, GRU num_layer, Dropout rate]
random_number_range=[(0.0001, 0.1), (1, 256), (1, 256), (1, 256), (1, 10), (0.1, 0.99)]
start_time = time.time()
X = [0.001, 64, 32, 3, 1, 0.5]

# train
start_time1 = time.time()
for wc in work_condition:
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    act_mse, best_mse ,train_result = Train_pipeline(train_vali, hyper_parameter, X, wc)
    model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
    torch.save(train_result, model_name)
    print("best MSE = {}".format(best_mse))
    print("{}, PTH saved done!".format(model_name))
end_time1 = time.time()
train_time = end_time1-start_time1

print("Train Finish !")
print("Train Time = {}".format(train_time))


# # test
# start_time2 = time.time()
# for wc in work_condition:
#     trained_pth = 'F:/git_repo/WKN_SSO/result/pth/'+exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st.pth'
#     if wc == 1:
#         Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_3',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_4',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_5',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_6',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_7']
#     elif wc == 2:
#         Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_3',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_4',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_5',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_6',
#                     'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_7']
#     elif wc == 3:
#         Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing3_3']
        
#     for test_set in Test_set:
#         tmp_result = []
#         bearing_name, tmp_result = Test_pipeline(trained_pth, test_set, hyper_parameter[0], wc)
#         output_2_csv(bearing_name, tmp_result)
#         output_2_plot(bearing_name, tmp_result)
# end_time2 = time.time()
# test_time = end_time2-start_time2

# print("Test Finish !")
# print("Test Time = {}".format(test_time))