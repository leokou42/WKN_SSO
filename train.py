
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import LA_WKN_BiGRU
from data_loader import Data

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

train_folder = './viberation_dataset/Learning_set/Bearing1_1/'
train_data = Data(train_folder)
X = train_data.get_hrz_data()
y = train_data.get_label()

test_folder = './viberation_dataset/Test_set/Bearing1_3/'
test_data = Data(test_folder)
X_test = test_data.get_hrz_data()
y_test = test_data.get_label()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 划分数据集为训练集、验证集和测试集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

# 将数据转换为PyTorch张量
X_train, y_train = torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device)
X_val, y_val = torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device)
X_test, y_test = torch.Tensor(X_test).to(device), torch.Tensor(y_test).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 创建并初始化WKN-BiGRU模型
model = LA_WKN_BiGRU().to(device)
# 可以加载预训练模型权重，如果有的话

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 根据您的任务选择合适的损失函数
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
num_epochs = 50  # 根据您的需求调整
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# 保存训练后的模型
torch.save(model.state_dict(), 'trained_model.pth')

# 模型评估
model.eval()
total_loss = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader)}")

# 在测试集上评估模型
total_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")
