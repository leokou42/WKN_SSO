
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import math

from utils import *
from models.ML_WKN_BiGRU_MSA import ML_WKN_BiGRU_MSA
from dataset_loader import CustomDataSet
from dataset_2stage import CustomDataSet_2stage


def Twostage_pipeline(Learning_Validation, hp, sX, work_condition):
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

    for acq in[1,2]:
        # setup experiment working condition and dataset location
        train_data = CustomDataSet_2stage(Learning_set, work_condition, acq_part=acq, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
        val_data = CustomDataSet_2stage(Validation_set, work_condition, acq_part=acq, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])

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
        # 改成k fold驗證方法

        # ===================================================================================================

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # model selection
        # model = CNN_GRU().to(device)
        model = ML_WKN_BiGRU_MSA(sX).to(device)

        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

        # print(sX)
        # 訓練模型
        best_MSE = 100
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
                    mse = criterion(outputs, labels)
                    # rmse = torch.sqrt(mse)
                    total_mse += mse.item() * labels.size(0)
                    num_samples += labels.size(0)

            average_mse = total_mse / num_samples
            act_MSE += average_mse
            all_loss.append(loss.cpu().detach().numpy())
            all_mse.append(average_mse)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation MSE: {average_mse:.4f}')

            if average_mse < best_MSE:
                best_MSE = average_mse
                if acq == 1:
                    best_model1 = model.state_dict()
                elif acq == 2:
                    best_model2 = model.state_dict()

        act_MSE = act_MSE / (epoch+1)
        print("Process MSE = {}".format(act_MSE))
        # print("Process RMSE = {}".format(act_MSE**2))
        # Lossplot = 'loss & MSE'+str(work_condition)
        # loss_2_plot(Lossplot, all_loss, all_mse)

    return act_MSE, best_MSE, best_model1, best_model2