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

def Test_pipeline(Test_set, batch_size, work_condition, show_pic = False):
    Bearing_name = Test_set.split('/')[-1]
    test_data = CustomDataSet(Test_set, work_condition, mode='test')
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = LA_WKN_BiGRU()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = LA_WKN_BiGRU().to(device)
    model.load_state_dict(torch.load('your_model.pth'))
    model.eval()

    print("target: {}".format(Test_set))
    ans = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            # print("size: {}".format(outputs.size()))
            # print(outputs)
            vals = outputs.tolist()
            for val in vals:
                ans.append(val)
    # print(ans)

    # 繪製圖形並儲存
    plt.plot(ans)
    plt.title(Bearing_name)
    plt.ylabel('Health Index')
    plt.xlabel('Time')
    pic_name = 'F:/git_repo/WKN_SSO/result/' + Bearing_name + '.png'
    plt.savefig(pic_name)
    if show_pic == True:
        plt.show()
        
    print("{} picture saved".format(Bearing_name))

    return ans