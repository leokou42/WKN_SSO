import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import CustomDataSet
from model import LA_WKN_BiGRU

work_condition = 1
batch_size = 32

Test_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_3'
# Test_set = "./viberation_dataset/Test_set/Bearing1_3"
Bearing_name = Test_set.split('/')[-1]
test_data = CustomDataSet(Test_set, work_condition, mode='test')
test_loader = DataLoader(test_data, batch_size=batch_size)

model = LA_WKN_BiGRU() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LA_WKN_BiGRU().to(device)
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

print("target: {}".format(Test_set))
temp_ans = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        print("size: {}".format(outputs.size()))
        # print(outputs)

        vals = outputs.tolist()
        temp_ans.append(vals)

ans = np.array(temp_ans).flatten().tolist()
# 繪製圖形
plt.plot(ans)
plt.title('Plot of the Data')
plt.xlabel('Health Index')
plt.ylabel('Time')
plt.show()

# 儲存圖形為圖片文件
pic_name = Bearing_name + '.png'
plt.savefig(pic_name)