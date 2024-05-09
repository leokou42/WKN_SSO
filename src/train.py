
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

def Train_pipeline(Learning_Validation, hp, sX, work_condition):
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
    
    # setup experiment working condition and dataset location
    train_data = CustomDataSet(Learning_set, work_condition, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
    val_data = CustomDataSet(Validation_set, work_condition, transform=None, mode='train')

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
    # model = CNN_BiGRU().to(device)
    model = ML_WKN_BiGRU_MSA(sX).to(device)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # print(sX)
    # 訓練模型
    best_MSE = 100
    act_MSE = 0
    all_loss = []
    all_mse = []
    early_stop = 0
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
        if average_mse > 0.05:
            early_stop += 1
        else:
            early_stop = 0
        all_loss.append(loss.cpu().detach().numpy())
        all_mse.append(average_mse)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation MSE: {average_mse:.4f}')
        # if early_stop > 9:
        #     print("underfitting, break")
        #     break
        if average_mse < best_MSE:
            best_MSE = average_mse
            best_model = model.state_dict()

    act_MSE = act_MSE / (epoch+1)
    print("Process MSE = {}".format(act_MSE))
    # print("Process RMSE = {}".format(act_MSE**2))
    # Lossplot = 'loss & MSE'+str(work_condition)
    # loss_2_plot(Lossplot, all_loss, all_mse)
    
    return act_MSE, best_MSE, best_model

if __name__ == "__main__":
    # noSSO training 
    # setup
    # random seed setup
    random.seed(42)
    np.random.seed(0)
    setup_seed(20)
    torch.cuda.empty_cache()
    hyper_parameter = [32,30]   # [batch_size, num_epochs]
    Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
    Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
    work_condition = [1]
    exp_topic = 'noSSO'
    exp_num = 0
    train_vali = [Learning_set, Validation_set, 3]

    # regular train
    # iX = [500, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 77, 83]
    iX = [[100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 50, 50],
          [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 60, 60],
          [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 70, 70],
          [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 80, 80],
          [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 90, 90] ]
    start_time1 = time.time()
    wc1 = []
    wc2 = []
    for term in range(5):
        for wc in work_condition:
            exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
            act_mse ,_ ,train_result = Train_pipeline(train_vali, hyper_parameter, iX[term], wc)
            if wc == 1:
                wc1.append(act_mse)
            elif wc == 2:
                wc2.append(act_mse)
            model_name = 'F:/git_repo/WKN_SSO/result/pth/' + exp_name + '.pth'
            # torch.save(train_result, model_name)
            print("{}, PTH saved done!".format(model_name))
        end_time1 = time.time()
        train_time = end_time1-start_time1

        print("Train Finish !")
        print("Train Time = {}".format(train_time))

        print(f'\nEpoch {term + 1}/30')
        print(wc1, sum(wc1)/(term+1))
        print(wc2, sum(wc2)/(term+1))
    
    csv_name = 'ML_WKN_BiGRU_MSA_0.90.9'
    train_2_csv(csv_name, wc1, wc2)

