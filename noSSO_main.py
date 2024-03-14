import sys
sys.path.append("src")
import time 
import numpy as np
import random
import torch

from src.train import Train_pipeline
from src.test import Test_pipeline
from src.utils import *

# noSSO training 

# setup
hyper_parameter = [32,50]   # [batch_size, num_epochs]
batch_size = hyper_parameter[0]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set'
work_condition = [1,2]
exp_topic = 'noSSO'
exp_num = 5
train_vali = [Learning_set, Validation_set, 3]
iX = [0.001, 32, 64, 16, 32, 32, 3, 1, 0.5, 64, 0.3, 0.6, 0.6]

# train
start_time1 = time.time()
for wc in work_condition:
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    _ ,_ ,train_result = Train_pipeline(train_vali, hyper_parameter, iX, wc)
    model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
    torch.save(train_result, model_name)
    print("{}, PTH saved done!".format(model_name))
end_time1 = time.time()
train_time = end_time1-start_time1

print("Train Finish !")
print("Train Time = {}".format(train_time))

#test
start_time2 = time.time()
for wc in work_condition:
    trained_pth = 'F:/git_repo/WKN_SSO/result/pth/'+exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st.pth'
    tmp_result = []
    bearing_name, tmp_result = Test_pipeline(trained_pth, batch_size, iX, wc)

end_time2 = time.time()
test_time = end_time2-start_time2

print("Test Finish !")
print("Test Time = {}".format(test_time))