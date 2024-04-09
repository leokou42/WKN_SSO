
import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F

def Morlet(p):
    C = pow(pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
    return y

class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Morlet_fast, self).__init__()
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
        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))
        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))
        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()
        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250
        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding='same', dilation=1, bias=None, groups=1)

class ML_WKN_BiGRU_MSA(nn.Module):

    def __init__(self, sX):
        self.sX = sX
        super(ML_WKN_BiGRU_MSA, self).__init__()
        self.WKN = nn.Sequential(
            Morlet_fast(out_channels=self.sX[1], kernel_size=self.sX[2]),  # x_1, SSO update output channel, original = 32
                                                                            # x_2, SSO update kernel size, original = 64
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.sX[1], self.sX[3], kernel_size=self.sX[4], padding='same'),  # x_3, SSO update output channel, original = 16
                                                                                        # x_4, SSO update kernel size, original = 32
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.sX[3], self.sX[5], kernel_size=self.sX[6], padding='same'),  # x_5, SSO update output channel, original = 32
                                                                                        # x_6, SSO update kernel size, original = 3
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.BiGRU = nn.GRU(input_size=self.sX[5], hidden_size=8, num_layers=sX[7], bidirectional=False) # x_7, SSO update num_layers, original = 1

        self.MSA = nn.MultiheadAttention(embed_dim=8, num_heads=4, batch_first=True, dropout=self.sX[8]/100) 
        # x_8, SSO update dropout rate, original = 0.5

        self.FC = nn.Sequential(
            nn.Linear(8, 8),
            nn.Flatten(),
            nn.Linear(2560, self.sX[9]),        # x_9, SSO update nuneral num, original = 64
            nn.ReLU(),
            nn.Dropout(self.sX[10]/100),        # x_10, SSO update Dropout rate, original = 0.3
            nn.Linear(self.sX[9],1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # print("in: {}".format(x.shape))
        x = self.WKN(x)
        # print("WKN out: {}".format(x.shape))
        x = x.permute(0, 2, 1)
        x,_ = self.BiGRU(x)
        # print("GRU out: {}".format(x.shape))
        x,_ = self.MSA(x,x,x)
        # print("MSA out: {}".format(x.shape))
        x = self.FC(x)
        x = x.squeeze()
        return x

if __name__ == "__main__":
    # test
    def SSO_hp_trans(iX):
        iX[0] = iX[0]/10000
        iX[8] = iX[8]/100
        iX[10] = iX[10]/100
        iX[11] = iX[11]/100
        iX[12] = iX[12]/100

        return iX

    testi = torch.randn(32, 1, 2560).cuda()
    X = [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 50, 50]
    sX = SSO_hp_trans(X)
    model = ML_WKN_BiGRU_MSA(sX).cuda()

    testo = model(testi)

    print("testo out: {}".format(testo.shape))
