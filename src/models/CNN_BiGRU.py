
import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F

class CNN_GRU(nn.Module):

    def __init__(self):
        super(CNN_GRU, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=32, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 16, kernel_size=32, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.BiGRU = nn.GRU(input_size=32, hidden_size=8, num_layers=1, bidirectional=True)

        self.FC = nn.Sequential(
            nn.Flatten(0,-1),
            nn.Linear(5120, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.CNN(x)    
        # print(x.shape)    
        x = x.view(32, 320, 32)
        x,_ = self.BiGRU(x)
        x = self.FC(x)
        return x

