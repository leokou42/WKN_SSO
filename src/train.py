
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import LA_WKN_BiGRU
from dataset_loader import CustomDataSet

def Train_pipeline(Learning_set, hyper_parameter, work_condition, exp_name):
    # access to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameter setup
    batch_size = hyper_parameter[0]
    num_epochs = hyper_parameter[1]
    learning_rate = hyper_parameter[2]

    # setup experiment working condition and dataset location
    train_data = CustomDataSet(Learning_set, work_condition, mode='train')

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LA_WKN_BiGRU().to(device)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # 訓練模型
    act_MSE = 0
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
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
                total_mse += mse.item() * labels.size(0)
                num_samples += labels.size(0)

        average_mse = total_mse / num_samples
        act_MSE += average_mse
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation MSE: {average_mse:.4f}')
    
    act_MSE = act_MSE / epoch
    print("Process MSE = {}".format(act_MSE))

    # 保存模型
    model_name = exp_name + '.pth'
    torch.save(model.state_dict(), model_name)
    
    return model_name