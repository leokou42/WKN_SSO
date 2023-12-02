
# In[1]:
import sys
sys.path.append("src")
import time 
import numpy as np
import random
import torch

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
Nrun, Ngen, Nsol, Nvar = 30, 10, 5, 7
Cg = 0.1        # GBEST區間
Cp = 0.3        # PBEST區間
Cw = 0.6        # 前解區間
Njob = 2        # 表連續型參數個數
Njob2 = 4       # 表離散型參數個數

#[learning rate, LA kernel, conv1 kernel, conv2 kernel, GRU num_layer, Dropout rate]
random_number_range = [(0.0001, 0.1), (1, 256), (1, 256), (1, 256), (1, 10), (0.1, 0.99)]
start_time = time.time()
X, FX, pX, pF, gBest, genBest = [], [], [], [], 0, 0,
value_to_x_dict = {}
umax = np.array([0] * Njob, dtype=np.float64)
umin = np.array([1000] * Njob, dtype=np.float64)
# pX存gbest解pF存值
gen = 0


# In[]:
# train
start_time_SSO = time.time()
start_time1 = time.time()
for wc in work_condition:
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'

    for sol in range(Nsol):
        rX = generate_random_numbers(random_number_range)
        X.append(rX) # add to the X2.
        pX.append(rX) # add to the X2.
        # value表MSE
        value = Train_pipeline(Learning_set, hyper_parameter, rX, wc, exp_name)
        for job in range(Njob):
            if X[sol][job] > umax[job]:
                umax[job] = X[sol][job]
            if X[sol][job] < umin[job]:
                umin[job] = X[sol][job]
        FX.append(value)
        pF.append(value)
        if FX[sol] > FX[gBest]:
            gBest = sol
            pX[gBest] = rX
        print("gen&sol = {} {}, current sol = {}, fit = {}".format(gen,sol ,rX,value))
        value_to_x_dict[value] = (tuple(X[sol]),(gen,sol))
    
    for gen in range(1,Ngen+1):
        for sol in range(Nsol):
            job = -1
            #sso
            while job < Njob + Njob2 - 1:
                job += 1
                rnd2 = np.random.rand()
                if rnd2 < Cg:
                    X[sol][job] = pX[gBest][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        X[sol][job] = int(random_select(job))
                elif rnd2 < Cp:
                    X[sol][job] = pX[sol][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        X[sol][job] = int(random_select(job))
                elif rnd2 < Cw:
                    X[sol][job] = X[sol-1][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        X[sol][job] = int(random_select(job))
                else:
                    X[sol][job] = int(random_select(job))
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        X[sol][job] = int(random_select(job))
                #計算value
                ##value表目標值，F表函數，RX是你想更新的變數
                value = Train_pipeline(Learning_set, hyper_parameter, rX, wc, exp_name)
            
            #計算value結束
            value_to_x_dict[value] = (tuple(X[sol]),(gen,sol))
            #判斷大小
            print("目前gbest",pF[gBest])
            if value > pF[sol]:
                pF[sol] = value
                if value > pF[gBest]:
                    gBest = sol
                    genBest = gen
                    # torch.save(model.state_dict(), model_name)
                    # print("better than gbest, value = {}".format(value))
            if gen == 1 and sol == 0:
                # torch.save(model.state_dict(), model_name)
                print("initial sol, value = {}".format(value))
    end_time_SSO = time.time()
    sso_time=end_time_SSO-start_time_SSO
    print("\n==============================================================================================")
    print("optimal sequence:", value_to_x_dict[pF[gBest]][0])
    print("optimal value: {}".format(pF[gBest]))
    print("optimal generation number:", value_to_x_dict[pF[gBest]][1][0])
    print("optimal sol number:", value_to_x_dict[pF[gBest]][1][1])
    print("sso_time:", sso_time)

    print("{}, PTH saved done!".format(model_name))

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