
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import LA_WKN_BiGRU
from dataset_loader import CustomDataSet

# hyperparameter setup
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# setup experiment working condition and dataset location
work_condition = 1
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
Test_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/'
# Learning_set = "./viberation_dataset/Learning_set/"
# Test_set = "./viberation_dataset/Test_set/"

train_data = CustomDataSet(Learning_set, work_condition, mode='train')
test_data = CustomDataSet(Test_set, work_condition, mode='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

model = LA_WKN_BiGRU().to(device)

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# 訓練模型
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
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation MSE: {average_mse:.4f}')

# 保存模型
torch.save(model.state_dict(), 'your_model.pth')

