
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
    
    # setup experiment working condition and dataset location
    train_data1 = CustomDataSet_2stage(Learning_set, work_condition, acq_part=1, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
    train_data2 = CustomDataSet_2stage(Learning_set, work_condition, acq_part=2, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
    val_data1 = CustomDataSet_2stage(Validation_set, work_condition, acq_part=1, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])
    val_data2 = CustomDataSet_2stage(Validation_set, work_condition, acq_part=2, transform=None, mode='train', label_style=2, two_stage_hp=[sX[11]/100, sX[12]/100])

    # ===================================================================================================
    # 不同的validation方法:
    train_size = int(0.8 * len(train_data1))
    val_size = len(train_data1) - train_size
    if Validation_type == 1:    # 隨機抽樣
        train_dataset, val_dataset = torch.utils.data.random_split(train_data1, [train_size, val_size])
    elif Validation_type == 2:    # 照順序抽樣
        train_dataset = torch.utils.data.Subset(train_data1, range(train_size))
        val_dataset = torch.utils.data.Subset(train_data1, range(train_size, train_size + val_size))
    elif Validation_type == 3:    # 使用不同的軸承資料做驗證
        train_dataset1 = train_data1
        train_dataset2 = train_data2
        val_dataset1 = val_data1
        val_dataset2 = val_data2
    # 改成k fold驗證方法

    # ===================================================================================================

    train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    val_loader1 = DataLoader(val_dataset1, batch_size=batch_size)
    val_loader2 = DataLoader(val_dataset2, batch_size=batch_size)

    # model selection
    # model = CNN_GRU().to(device)
    model = ML_WKN_BiGRU_MSA(sX).to(device)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # print(sX)
    # 訓練模型
