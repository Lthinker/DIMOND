import torch.nn as nn
import torch
import numpy as np
class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)

def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor w0
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p
        
    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x
        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,nl):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding=0,padding_mode='zeros')
        self.relu = nl
        
    def forward(self, x):
        x  # 保存输入特征
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = out + x  # 残差连接：将输入特征与输出特征相加
        out = self.relu(out)
        
        return out

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels,out_channels,nl):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding=0,padding_mode='zeros')
        self.relu1 = nl
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1,padding=0,padding_mode='zeros')
        self.relu2 = nl
        
    def forward(self, x):
        x  # 保存输入特征
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        out = out + x  # 残差连接：将输入特征与输出特征相加
        out = self.relu2(out)
        
        return out