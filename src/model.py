
import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F

def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Laplace_fast, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):
        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        p1 = time_disc.unsqueeze(0) - self.b_ / self.a_
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size)
        # print(waveforms.shape)
        # waveforms = waveforms.squeeze()
        return F.conv1d(waveforms, self.filters, stride=1, padding='same', dilation=1, bias=None, groups=1)

class LA_WKN_BiGRU(nn.Module):

    def __init__(self):
        super(LA_WKN_BiGRU, self).__init__()
        self.WKN = nn.Sequential(
            Laplace_fast(out_channels=32, kernel_size=64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 16, kernel_size=32, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.BiGRU = nn.GRU(input_size=32, hidden_size=8, num_layers=1, bidirectional=True)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5120, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.WKN(x)    
        # print(x.shape)    
        x = x.permute(0, 2, 1)
        x,_ = self.BiGRU(x)
        # print(x.shape)
        x = self.FC(x)
        x = x.squeeze()
        return x

