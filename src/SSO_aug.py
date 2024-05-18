
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
    X, FX, pX, pF, gBest= [], [], [], [], 0,
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
                if rnd2 < Cg:
                    X[sol][job] = pX[gBest][job]
                    if X[sol][job]<random_number_range[job][0] or X[sol][job]>random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
                elif rnd2 < Cp:
                    X[sol][job] = pX[sol][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
                elif rnd2 < Cw:
                    X[sol][job] = X[sol-1][job]
                    if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                        if X[sol][job] > 1:
                            X[sol][job] = int(random_select(job, random_number_range))
                        else:
                            X[sol][job] = random_select(job, random_number_range)
                else:
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

if __name__ == "__main__":
    # setup
    hyper_parameter = [32, 30]   # [batch_size, num_epochs]
    Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
    Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
    work_condition = [1, 2]
    exp_topic = 'Final_SSO_ML'
    exp_num = 30

    random_number_range=[(1, 1000),     # learning rate     0
                        (1, 64),        # LA kernel num     1
                        (2, 128),       # LA kernel size    2
                        (1, 64),        # Conv1 num         3
                        (2, 32),        # Conv1 size        4
                        (1, 64),        # Conv2 num         5
                        (2, 32),        # Conv2 size        6
                        (1, 10),        # Gru layers        7
                        (1, 99),        # MSA dropout       8
                        (1, 5120),      # Linear nuron nums 9
                        (1, 99),        # dropout           10
                        (50, 90),       # label twist_point 11
                        (50, 90)]       # label slope       12

    iX = [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 60, 60]
    start_time = time.time()

    # SSO setup
    Cg = [0.7, 0.1, 0.1, 0.1]  #GBEST區間
    Cp = [0.8, 0.8, 0.2, 0.2]  #PBEST區間
    Cw = [0.9, 0.9, 0.9, 0.3]  #前解區間
    Nsol = 5
    Ngen = 10

    # train
    start_time1 = time.time()
    train_vali = [Learning_set, Validation_set, 3]
    num_c = 1
    for term in range(exp_num):
        print("Current terms = {}".format(term+1))
        for c in range(num_c):
            for wc in work_condition:
                train_detail = [train_vali, hyper_parameter, wc]
                print("start SSO !")
                print(train_detail)
                exp_name = exp_topic+'_wc'+str(wc)+'_'+str(term+1)+'st'
                iX = [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 60, 60]
                print("Cg = {}, Cp = {}, Cw = {}".format(Cg[c], Cp[c], Cw[c]))
                all_result = SSO_train(exp_name, Cg[c], Cp[c], Cw[c], Nsol, Ngen, random_number_range, iX, train_detail)
                min_key, min_value = find_min_key_value(all_result)
                print("min_key = {} , value of {}".format(min_key, min_value))
                csv_name = 'F:/git_repo/WKN_SSO/result/SSO_result/'+exp_topic+'_wc'+str(wc)+'_'+str(term+1)+'st'+'_vali'+str(train_vali[2])+'_'+'SSOHP_setup_'+str(c+1)+'st.csv'
                print(all_result)
                SSO_2_csv(csv_name, all_result)

            end_time1 = time.time()
            train_time = end_time1-start_time1

            print("SSO Train Finish !")
            print("SSO Train Time = {}".format(train_time))

