import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import CustomDataSet
from model import LA_WKN_BiGRU

model = LA_WKN_BiGRU()  # 请替换为你的模型类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LA_WKN_BiGRU().to(device)
model.load_state_dict(torch.load('your_model.pth'))  # 请替换为你的模型.pth文件的路径
model.eval()  # 切换模型为评估模式

# 准备输入数据（震动数据），这里使用示例数据，你需要根据你的数据结构进行调整
input_path = '/Users/yentsokuo/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_3/acc_00001.csv'  # 一个(1, 2560)尺寸的震动数据
input_data = pd.read_csv(input_path)
input_tensor = torch.Tensor(input_data)

# 使用模型进行预测
with torch.no_grad():
    predicted_remaining_life = model(input_tensor)

# 处理预测结果，这取决于你的任务
# 在这个例子中，predicted_remaining_life是一个float，表示预测的剩余寿命
print(f'Predicted Remaining Life: {predicted_remaining_life.item()}')