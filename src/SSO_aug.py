
import numpy as np
import time
import random
import torch
import torch.nn as nn
import numpy as np
import time

from utils import *
from train import Train_pipeline

def SSO_train(exp_name, Cg, Cp, Cw, Nsol, Ngen, random_number_range, initial_sol, train_detail):
    random.seed(42)
    start_time = time.time()
    print("start SSO train at {}".format(start_time))

    model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
    train_vali = train_detail[0]
    hyper_parameter = train_detail[1]
    wc = train_detail[2]

    Nvar = len(random_number_range)
    gen = 0
    X, FX, pX, pF, gBest, genBest= [], [], [], [], 0, 0,
    value_to_x_dict = {}
    X.append(initial_sol)
    pX.append(initial_sol)
    _,initial_fit,_ =  Train_pipeline(train_vali, hyper_parameter, initial_sol, wc)
    FX.append(initial_fit)
    pF.append(initial_fit)
    value_to_x_dict[initial_fit] = (tuple(X[0]), (gen, 0))
    print(gen, 0, initial_sol, initial_fit)
    print("=====================================")

    for sol in range(Nsol):
        if sol != 0:
            rX = generate_random_numbers(random_number_range)
            print("(gen, sol) = ({}, {})".format(gen, sol))
            print("result X = {}".format(rX))
            X.append(rX)
            pX.append(rX)
            _,fit,_ = Train_pipeline(train_vali, hyper_parameter,rX, wc)
            FX.append(fit)
            pF.append(fit)
            print("MSE fit = {}".format(fit))
            value_to_x_dict[fit] = (tuple(X[sol]),(gen,sol))
        if FX[sol] < FX[gBest]:
            gBest = sol
            pX[gBest] = rX

    for gen in range(1, Ngen+1):
        for sol in range(Nsol):
            job=-1

            # SSO 
            while job < Nvar-1:
                job += 1
                rnd2 = np.random.rand()
                stat = 0
                if rnd2 < Cg:
                    stat = 1
                    X[sol][job] = pX[gBest][job]
                    if X[sol][job]<random_number_range[job][0] or X[sol][job]>random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
                elif rnd2 < Cp:
                    stat = 2
                    X[sol][job] = pX[sol][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
                elif rnd2 < Cw:
                    stat = 3
                    X[sol][job] = X[sol-1][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
                else:
                    stat = 4
                    X[sol][job] = int(random_select(job, random_number_range))
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
            print("(gen, sol) = ({}, {})".format(gen, sol))
            print("result X = {}".format(X[sol]))
            _, fit, train_result =  Train_pipeline(train_vali, hyper_parameter, X[sol], wc)
            print("MSE fit = {}".format(fit))
            value_to_x_dict[fit] = (tuple(X[sol]),(gen,sol))
            if fit < pF[sol]:
                pF[sol] = fit
                if fit < pF[gBest]:
                    gBest = sol
                    torch.save(train_result, model_name)
            if gen == 1 and sol == 0:
                torch.save(train_result, model_name)
    
    end_time = time.time()
    sso_time = end_time - start_time
    print("optimal sequence",value_to_x_dict[pF[gBest]][0])
    print("optimal value:{}".format(pF[gBest]))
    print("optimal generation number:", value_to_x_dict[pF[gBest]][1][0])
    print("optimal sol number:", value_to_x_dict[pF[gBest]][1][1])
    print("sso_time:", sso_time)

    return value_to_x_dict


# setup
hyper_parameter = [32, 15]   # [batch_size, num_epochs]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
work_condition = [1,2]
exp_topic = 'SSO'
exp_num = 5

#[learning rate, LA kernel, conv1 kernel, conv2 kernel, GRU num_layer, MSA Dropout rate, FC Dropout rate, label twist_point, label slope]
random_number_range=[(0.0001, 0.1), (1, 64), (1, 64), (1, 32), (1, 10), (0.1, 0.99), (0.1, 0.99), (0.5, 0.8), (0.5, 0.8)]
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
for wc in work_condition:
    train_detail = [train_vali, hyper_parameter, wc]
    print("start SSO !")
    print(train_detail)
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    all_result = SSO_train(exp_name, Cg, Cp, Cw, Nsol, Ngen, random_number_range, iX, train_detail)
    min_key, min_value = find_min_key_value(all_result)
    print("min_key = {} , value of {}".format(min_key, min_value))
    csv_name = 'F:/git_repo/WKN_SSO/result/SSO_result/'+exp_topic+'_wc'+str(wc)+'_vali'+str(train_vali[2])+'_'+str(exp_num)+'st.csv'
    SSO_2_csv(csv_name, all_result)

end_time1 = time.time()
train_time = end_time1-start_time1

print("Train Finish !")
print("Train Time = {}".format(train_time))

