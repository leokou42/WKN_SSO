
import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F

class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 16, kernel_size=32, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.LSTM = nn.LSTM(input_size=32, hidden_size=8, num_layers=1, bidirectional=False)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # print("in: {}".format(x.shape))
        x = self.CNN(x)    
        # print("CNN out: {}".format(x.shape))
        x = x.permute(0, 2, 1)
        x,_ = self.LSTM(x)
        # print("LSTM out: {}".format(x.shape))
        x = self.FC(x)
        x = x.squeeze()
        return x

if __name__ == "__main__":
    # test
    testi = torch.randn(32, 1, 2560)
    model = CNN_LSTM()

    testo = model(testi)

    print("testo out: {}".format(testo.shape))