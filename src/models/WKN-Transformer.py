
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

class TransformerModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_prob):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout_prob,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  
        x = self.sigmoid(x)  

        return x
    
class WKN_Transformer(nn.Module):

    def __init__(self, sX):
        self.sX = sX
        super(WKN_Transformer, self).__init__()
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

        self.MSA = nn.Sequential(

        )

if __name__ == "__main__":
    testi = torch.randn(32, 1, 2560).cuda()

    model = Laplace_fast(32, 64).cuda()

    testo = model(testi)
    print("testo out: {}".format(testo.shape))