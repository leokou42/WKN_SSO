
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import math

from utils import *
from models.LA_WKN_BiGRU import LA_WKN_BiGRU
from dataset_loader import CustomDataSet

def Train_pipeline(Learning_set, hp, X, work_condition):
    # access to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameter setup
    batch_size = hp[0]
    num_epochs = hp[1]
    
    learning_rate = X[0] # SSO update learning_rate, original = 0.001

    # setup experiment working condition and dataset location
    train_data = CustomDataSet(Learning_set, work_condition, mode='train')

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    # # 隨機抽樣
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
    # 照順序抽樣
    # train_dataset = torch.utils.data.Subset(train_data, range(train_size))
    # val_dataset = torch.utils.data.Subset(train_data, range(train_size, train_size + val_size))
    
    # 改成k fold驗證方法
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LA_WKN_BiGRU(X).to(device)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # 訓練模型
    act_MSE = 0
    all_loss = []
    all_mse = []
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            # loss = torch.sqrt(criterion(outputs, labels))
            loss.backward()
            optimizer.step()

        # 在驗證集上評估模型
        model.eval()
        total_mse = 0
        num_samples = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                # mse = criterion(outputs, labels)
                rmse = torch.sqrt(criterion(outputs, labels))
                # total_mse += mse.item() * labels.size(0)
                total_mse += rmse.item() * labels.size(0)
                num_samples += labels.size(0)

        average_mse = total_mse / num_samples
        act_MSE += average_mse
        all_loss.append(loss)
        all_mse.append(average_mse)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation MSE: {average_mse:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation RMSE: {average_mse:.4f}')
    
    act_MSE = act_MSE / epoch
    print("Process MSE = {}".format(act_MSE))
    print("Process RMSE = {}".format(act_MSE**2))
    # loss_2_plot(exp_name, all_loss, all_mse)


    # 保存模型
    # model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
    # torch.save(model.state_dict(), model_name)
    
    return act_MSE, model.state_dict()

def SSO_train():
    # SSO setup
    Ngen, Nsol = 2, 2
    Cg = 0.1        # GBEST區間
    Cp = 0.3        # PBEST區間
    Cw = 0.6        # 前解區間
    Njob = 2        # 表連續型參數個數
    Njob2 = 4       # 表離散型參數個數

    #[learning rate=0.001, LA kernel=64 , conv1 kernel=32, conv2 kernel=3, GRU num_layer=1, Dropout rate=0.5]
    random_number_range = [(0.0001, 0.01), (1, 256), (1, 64), (1, 32), (1, 10), (0.1, 0.99)]
    start_time = time.time()
    X, FX, pX, pF, gBest, genBest = [], [], [], [], 0, 0,
    value_to_x_dict = {}
    umax = np.array([0] * Njob, dtype=np.float64)
    umin = np.array([1000] * Njob, dtype=np.float64)
    # pX存gbest解pF存值
    gen = 0

    start_time_SSO = time.time()
    for wc in work_condition:
        exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
        model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'

        for sol in range(Nsol):
            rX = generate_random_numbers(random_number_range)
            X.append(rX) # add to the X2.
            pX.append(rX) # add to the X2.
            # value表MSE
            print("X = {}".format(X))
            value, model_p = Train_pipeline(Learning_set, hyper_parameter, rX, wc, exp_name)
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
                    print('random num ={}'.format(rnd2))
                    if rnd2 < Cg:
                        X[sol][job] = pX[gBest][job]
                        if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                            X[sol][job] = int(random_select(job, random_number_range))
                    elif rnd2 < Cp:
                        X[sol][job] = pX[sol][job]
                        if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                            X[sol][job] = int(random_select(job, random_number_range))
                    elif rnd2 < Cw:
                        X[sol][job] = X[sol-1][job]
                        if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                            X[sol][job] = int(random_select(job, random_number_range))
                    else:
                        X[sol][job] = int(random_select(job, random_number_range))
                        if X[sol][job] < random_number_range[job][0] or X[sol][job] > random_number_range[job][1]:
                            X[sol][job] = int(random_select(job, random_number_range))
                    #計算value
                    ##value表目標值，F表函數，RX是你想更新的變數
                    value, model_p= Train_pipeline(Learning_set, hyper_parameter, rX, wc, exp_name)
                
                #計算value結束
                value_to_x_dict[value] = (tuple(X[sol]),(gen,sol))
                #判斷大小
                print("目前gbest",pF[gBest])
                if value < pF[sol]:
                    pF[sol] = value
                    if value < pF[gBest]:
                        gBest = sol
                        genBest = gen
                        torch.save(model_p, model_name)
                        print("better than gbest, value = {}".format(value))
                if gen == 1 and sol == 0:
                    torch.save(model_p, model_name)
                    print("initial sol, value = {}".format(value))
        end_time_SSO = time.time()
        sso_time = end_time_SSO - start_time_SSO
        print("\n==============================================================================================")
        print("optimal sequence:", value_to_x_dict[pF[gBest]][0])
        print("optimal value: {}".format(pF[gBest]))
        print("optimal generation number:", value_to_x_dict[pF[gBest]][1][0])
        print("optimal sol number:", value_to_x_dict[pF[gBest]][1][1])
        print("sso_time:", sso_time)

        print("{}, PTH saved done!".format(model_name))

# setup
# hyper_parameter = [32, 50]   # [batch_size, num_epochs]
# Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
# work_condition = [1,2]
# exp_topic = 'noSSO'
# exp_num = 2

# # regular train
# X = [0.001, 64, 32, 3, 1, 0.5]
# start_time1 = time.time()
# for wc in work_condition:
#     exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
#     _ ,train_result = Train_pipeline(Learning_set, hyper_parameter, X, wc, exp_name)
#     model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
#     torch.save(train_result, model_name)
#     print("{}, PTH saved done!".format(model_name))
# end_time1 = time.time()
# train_time = end_time1-start_time1

# print("Train Finish !")
# print("Train Time = {}".format(train_time))

# SSO_train()