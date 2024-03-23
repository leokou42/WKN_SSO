
import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F

def Laplace(p, A, ep, tal, f):
    # A = 0.08
    # ep = 0.03
    # tal = 0.1
    # f = 50
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
        p1 = time_disc.unsqueeze(0).cuda() - self.b_.cuda()/ self.a_.cuda()
        laplace_filter = Laplace(p1, A = 0.08, ep = 0.03, tal = 0.1, f = 50)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()
        # print(waveforms.shape)
        # waveforms = waveforms.squeeze()
        return F.conv1d(waveforms, self.filters, stride=1, padding='same', dilation=1, bias=None, groups=1)

class LA_WKN_BiGRU(nn.Module):

    def __init__(self, sX):
        self.sX = sX
        super(LA_WKN_BiGRU, self).__init__()
        self.WKN = nn.Sequential(
            Laplace_fast(out_channels=self.sX[1], kernel_size=self.sX[2]),  # x_1, SSO update output channel, original = 32
                                                                            # x_2, SSO update kernel size, original = 64
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.sX[1], self.sX[3], kernel_size=self.sX[4], padding='same'),  # x_3, SSO update output channel, original = 16
                                                                                        # x_4, SSO update kernel size, original = 32
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(self.sX[3], self.sX[5], kernel_size=self.sX[6], padding='same'),  # x_5, SSO update output channel, original = 32
                                                                                        # x_6, SSO update kernel size, original = 3
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.BiGRU = nn.GRU(input_size=self.sX[5], hidden_size=8, num_layers=sX[7], bidirectional=True) # x_7, SSO update num_layers, original = 1

        self.MSA = nn.MultiheadAttention(embed_dim=16, num_heads=8, batch_first=True, dropout=self.sX[8]/100) 
        # x_8, SSO update dropout rate, original = 0.5

        self.FC = nn.Sequential(
            nn.Linear(16, 16),
            nn.Flatten(),
            nn.Linear(5120, self.sX[9]),    # x_9, SSO update nuneral num, original = 64
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
        # x = x.permute(1, 0, 2)
        x = self.FC(x)
        x = x.squeeze()
        return x

'''
# test
def SSO_hp_trans(iX):
    iX[0] = iX[0]/10000
    iX[8] = iX[8]/100
    iX[10] = iX[10]/100
    iX[11] = iX[11]/100
    iX[12] = iX[12]/100

    return iX

testi = torch.randn(32, 1, 2560).cuda()
# X = [0.001, 64, 32, 3, 1, 8, 0.5, 0.5, 0.6, 0.6]
X = [267, 60, 64, 39, 6, 46, 6, 4, 97, 3277, 55, 77, 83]
sX = SSO_hp_trans(X)
model = LA_WKN_BiGRU(sX).cuda()

testo = model(testi)

print("testo out: {}".format(testo.shape))
'''
