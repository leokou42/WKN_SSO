
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import math

from utils import *
from models.ML_WKN_BiGRU_MSA import ML_WKN_BiGRU_MSA
from dataset_2stage import CustomDataSet_2stage


def Twostage_pipeline(Learning_Validation, hp, sX, work_condition):
    # setup random seeds
    random.seed(42)
    np.random.seed(0)
    setup_seed(20)
    # access to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # hyperparameter setup
    # sX = SSO_hp_trans(iX)
    batch_size = hp[0]
    num_epochs = hp[1]
    learning_rate = sX[0]/100000 # SSO update learning_rate, original = 0.001
    # learning_rate = 0.001 # SSO update learning_rate, original = 0.001
    Learning_set = Learning_Validation[0]
    Validation_set = Learning_Validation[1]
    Validation_type = Learning_Validation[2]

    for acq in[1, 2]:
        print('acq = {}'.format(acq))
        # setup experiment working condition and dataset location
        train_data = CustomDataSet_2stage(Learning_set, work_condition, acq_part=acq, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
        val_data = CustomDataSet_2stage(Validation_set, work_condition, acq_part=acq, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
        print(len(train_data), len(val_data))

        # ===================================================================================================
        # 不同的validation方法:
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        if Validation_type == 1:    # 隨機抽樣
            train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
        elif Validation_type == 2:    # 照順序抽樣
            train_dataset = torch.utils.data.Subset(train_data, range(train_size))
            val_dataset = torch.utils.data.Subset(train_data, range(train_size, train_size + val_size))
        elif Validation_type == 3:    # 使用不同的軸承資料做驗證
            train_dataset = train_data
            val_dataset = val_data

        # ===================================================================================================
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # model selection
        model = ML_WKN_BiGRU_MSA(sX).to(device)
        # print(sX)

        # training and validation
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

        best_MSE = 100
        act_MSE = 0
        all_loss = []
        all_mse = []
        # all_mse1 = []
        # all_mse2 = []
        for epoch in range(num_epochs):
            model.train()
            for data, labels in train_loader:
                data, labels = data.float().to(device), labels.float().to(device)
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
                    data, labels = data.float().to(device), labels.float().to(device)
                    outputs = model(data)
                    mse = criterion(outputs, labels)
                    # rmse = torch.sqrt(mse)
                    total_mse += mse.item() * labels.size(0)
                    num_samples += labels.size(0)

            average_mse = total_mse / num_samples
            act_MSE += average_mse
            all_loss.append(loss.cpu().detach().numpy())
            all_mse.append(average_mse)
            # if acq == 1:
            #     all_mse1.append(average_mse)
            # elif acq == 2:
            #     all_mse2.append(average_mse)

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation MSE: {average_mse:.4f}')

            if average_mse < best_MSE:
                best_MSE = average_mse
                best_model = model.state_dict()
                # if acq == 1:
                #     best_model1 = model.state_dict()
                # elif acq == 2:
                #     best_model2 = model.state_dict()

        act_MSE = act_MSE / (num_epochs)
        print("Process MSE = {}".format(act_MSE))
        # print("Process RMSE = {}".format(act_MSE**2))
        # Lossplot = 'loss & MSE'+str(work_condition)
        # loss_2_plot(Lossplot, all_loss, all_mse)

    return act_MSE, best_MSE, best_model

if __name__ == "__main__":
    # noSSO 2stage training 
    # setup
    hyper_parameter = [32,30]   # [batch_size, num_epochs]
    Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
    Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
    work_condition = [1,2]
    exp_topic = '2stage'
    exp_num = 0
    train_vali = [Learning_set, Validation_set, 3]

    # regular train
    iX = [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 60, 60]
    start_time1 = time.time()

    for wc in work_condition:
        print('wc = {}'.format(wc))
        exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
        act_mse, _, train_result = Twostage_pipeline(train_vali, hyper_parameter, iX, wc)
        model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + 'part1.pth'
        torch.save(train_result, model_name)
        # torch.save(train_result1, model_name)
        # torch.save(train_result2, model_name)
        print("{}, PTH saved done!".format(model_name))
        print('work condition'+ str(wc)+ ': '+ str(act_mse))
    end_time1 = time.time()
    train_time = end_time1-start_time1

    print("Train Finish !")
    print("Train Time = {}".format(train_time))