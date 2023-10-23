
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
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
        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()
        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


class WKN(nn.modules):
    def __init__(self):
        super(WKN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class BiGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_neurons):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_neurons)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

class LA_WKN_LSTM(nn.Module):

    def __init__(self):
        super(LA_WKN_LSTM, self).__init__()
        self.args = args
        self.features = nn.Sequential(
            Laplace_fast(64, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=32, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.lstm = nn.LSTM(input_size=args.out_channels, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.output_size)




# ===============================================================================================
class CNN_LSTM(nn.Module):

    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.in_channels, out_channels=args.out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=args.out_channels, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x[:, -1, :]

        return x
