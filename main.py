
# In[1]:
import sys
sys.path.append("src")
import time 
import numpy as np
import random

from src.train import Train_pipeline
from src.test import Test_pipeline
from src.utils import *


# In[2]:
# setup
hyper_parameter = [32, 50]   # [batch_size, num_epochs]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
work_condition = [1,2,3]
exp_topic = 'noSSO'
exp_num = 1

# In[3]:
# SSO setup
Cg = 0.1        # GBEST區間
Cp = 0.3        # PBEST區間
Cw = 0.6        # 前解區間
Njob = 2        # 表連續型參數個數
Njob2 = 4       # 表離散型參數個數

#[learning rate, LA kernel, conv1 kernel, conv2 kernel, GRU num_layer, Dropout rate]
random_number_range=[(0.0001, 0.1), (1, 256), (1, 256), (1, 256), (1, 10), (0.1, 0.99)]
start_time = time.time()
X, FX, pX, pF, gBest, genBest = [], [], [], [], 0, 0,
value_to_x_dict = {}
umax=np.array([0]*Njob, dtype=np.float64)
umin=np.array([1000]*Njob, dtype=np.float64)
# pX存gbest解pF存值
gen=0
def random_select(n):
    return random.uniform(random_number_range[n][0],random_number_range[n][1])


# In[]:
# train
start_time1 = time.time()
for wc in work_condition:
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    train_result = Train_pipeline(Learning_set, hyper_parameter, X, wc, exp_name)
    model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
    print("{}, PTH saved done!".format(train_result))
end_time1 = time.time()
train_time = end_time1-start_time1

print("Train Finish !")
print("Train Time = {}".format(train_time))


# In[]:
# test
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