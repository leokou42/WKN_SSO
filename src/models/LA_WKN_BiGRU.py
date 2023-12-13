
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
        p1 = time_disc.unsqueeze(0).cuda() - self.b_.cuda()/ self.a_.cuda()
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()
        # print(waveforms.shape)
        # waveforms = waveforms.squeeze()
        return F.conv1d(waveforms, self.filters, stride=1, padding='same', dilation=1, bias=None, groups=1)

class LA_WKN_BiGRU(nn.Module):

    def __init__(self, X):
        self.X = X
        super(LA_WKN_BiGRU, self).__init__()
        self.WKN = nn.Sequential(
            Laplace_fast(out_channels=32, kernel_size=X[1]),  # SSO update kernel size, original = 64
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 16, kernel_size=X[2], padding='same'), # SSO update kernel size, original = 32
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=X[3], padding='same'), # SSO update kernel size, original = 3
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.BiGRU = nn.GRU(input_size=32, hidden_size=8, num_layers=X[4], bidirectional=True) #SSO update num_layers, original = 1

        self.MSA = nn.MultiheadAttention(embed_dim=16, num_heads=8)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5120, 64),
            nn.ReLU(),
            nn.Dropout(X[5]), # SSO update Dropout rate, original = 0.5
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.WKN(x)
        x = x.permute(2, 0, 1)
        x,_ = self.BiGRU(x)
        print("GRU out: {}".format(x.shape))
        x = x.transpose(0, 1)
        x,_ = self.MSA(x,x,x)
        print("MSA out: {}".format(x.shape))
        x = self.FC(x)
        x = x.squeeze()
        return x

'''
def forward(self, x):
    x = self.WKN(x)    
    # print(x.shape)
    # if x.shape == torch.Size([32, 32, 320]):
    #     x = x.permute(0, 2, 1)
    # elif x.shape == torch.Size([32, 320]):
    #     x = x.permute(1, 0)
    x = x.permute(0, 2, 1)
    x,_ = self.BiGRU(x)
    # print(x.shape)
    x = self.FC(x)
    x = x.squeeze()
    return x
'''

# test
testi = torch.randn(32, 1, 2560).cuda()
X = [0.001, 64, 32, 3, 1, 0.5]
model = LA_WKN_BiGRU(X).cuda()

testo = model(testi)

print("testo out: {}".format(testo.shape))

