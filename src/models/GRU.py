import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F

class GRU(nn.Module):

    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=8, num_layers=1, bidirectional=False)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20480, 5120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(5120, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # print("in: {}".format(x.shape))
        x = x.permute(0, 2, 1)
        x,_ = self.gru(x)
        # print("GRU out: {}".format(x.shape))
        x = self.FC(x)
        x = x.squeeze()
        return x

if __name__ == "__main__":
    # test
    testi = torch.randn(32, 1, 2560)
    model = GRU()

    testo = model(testi)

    print("testo out: {}".format(testo.shape))