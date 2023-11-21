
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import CustomDataSet
from model import LA_WKN_BiGRU
from utils import *

work_condition = 1
batch_size = 32

def Test_pipeline(trained_pth, Test_set, batch_size, work_condition):
    Bearing_name = Test_set.split('/')[-1]
    test_data = CustomDataSet(Test_set, work_condition, mode='test')
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = LA_WKN_BiGRU()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = LA_WKN_BiGRU().to(device)
    model.load_state_dict(torch.load(trained_pth))
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

    return Bearing_name, ans