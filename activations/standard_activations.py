import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    """Standard ReLU activation function"""
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return F.relu(x, inplace=self.inplace)

class LeakyReLU(nn.Module):
    """Leaky ReLU activation function"""
    def __init__(self, negative_slope=0.01, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope, inplace=self.inplace)

class ELU(nn.Module):
    """Exponential Linear Unit (ELU) activation function"""
    def __init__(self, alpha=1.0, inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self, x):
        return F.elu(x, self.alpha, inplace=self.inplace)

class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function"""
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)
