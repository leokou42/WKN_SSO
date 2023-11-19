import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import CustomDataSet
from model import LA_WKN_BiGRU

work_condition = 1
batch_size = 32

# Test_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/'
Test_set = "./viberation_dataset/Test_set/"
test_data = CustomDataSet(Test_set, work_condition, mode='test')
test_loader = DataLoader(test_data, batch_size=batch_size)

model = LA_WKN_BiGRU()  # 请替换为你的模型类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LA_WKN_BiGRU().to(device)
model.load_state_dict(torch.load('your_model.pth'))  # 请替换为你的模型.pth文件的路径
model.eval()  # 切换模型为评估模式

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)

print(outputs)
# with torch.no_grad():
#     predicted_remaining_life = model(input_tensor)

# 处理预测结果，这取决于你的任务
# 在这个例子中，predicted_remaining_life是一个float，表示预测的剩余寿命
# print(f'Predicted Remaining Life: {predicted_remaining_life.item()}')