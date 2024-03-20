
import sys
sys.path.append("src")
import time 
import numpy as np
import random
import torch

# from src.train import Train_pipeline
from src.test import Test_pipeline
from src.SSO_aug import SSO_train
from src.utils import *

# setup
hyper_parameter = [32, 15]   # [batch_size, num_epochs]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
work_condition = [1,2]
exp_topic = 'SSO'
exp_num = 4

#[learning rate, LA kernel, conv1 kernel, conv2 kernel, GRU num_layer, MSA heads, MSA Dropout rate, FC Dropout rate, label twist_point, label slope]
random_number_range=[(0.0001, 0.1), (1, 256), (1, 256), (1, 256), (1, 10), (0.1, 0.99), (0.1, 0.99), (0.5, 0.99), (0.5, 0.99)]
iX = [0.001, 64, 32, 3, 1, 0.5, 0.3, 0.6, 0.6]
start_time = time.time()

# SSO setup
Cg = 0.1  #GBEST區間
Cp = 0.3  #PBEST區間
Cw = 0.6  #前解區間
Nsol = 3
Ngen = 2

# train
# setup
hyper_parameter = [32, 20]   # [batch_size, num_epochs]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
work_condition = [1,2]
exp_topic = 'SSO'
exp_num = 4

#[learning rate, LA kernel, conv1 kernel, conv2 kernel, GRU num_layer, MSA heads, MSA Dropout rate, FC Dropout rate, label twist_point, label slope]
random_number_range=[(0.0001, 0.1), (32, 256), (1, 64), (1, 32), (1, 10), (0.1, 0.99), (0.1, 0.99), (0.5, 0.99), (0.5, 0.99)]
iX = [0.001, 64, 32, 3, 1, 0.5, 0.3, 0.6, 0.6]
start_time = time.time()

# SSO setup
Cg = 0.1  #GBEST區間
Cp = 0.3  #PBEST區間
Cw = 0.6  #前解區間
Nsol = 6
Ngen = 10

# train
start_time1 = time.time()
train_vali = [Learning_set, Validation_set, 1]
result_dict = {}
for wc in work_condition:
    train_detail = [train_vali, hyper_parameter, wc]
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    all_result = SSO_train(exp_name, Cg, Cp, Cw, Nsol, Ngen, random_number_range, iX, train_detail)
    min_key, min_value = find_min_key_value(all_result)
    print("min_key = {} , value of {}".format(min_key, min_value))

end_time1 = time.time()
train_time = end_time1-start_time1

print("Train Finish !")
print("Train Time = {}".format(train_time))
print(result_dict)

# # test
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