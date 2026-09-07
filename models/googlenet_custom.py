import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation

class Inception(nn.Module):
    """Inception module for GoogLeNet"""
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, activation='relu'):
        super(Inception, self).__init__()
        self.activation = get_activation(activation)

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
        )
        self.b2_3x3 = nn.Sequential(
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
        )

        # 1x1 conv -> 5x5 conv branch (using two 3x3 convs instead of 5x5)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
        )
        self.b3_5x5_1 = nn.Sequential(
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
        )
        self.b3_5x5_2 = nn.Sequential(
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_conv = nn.Sequential(
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
        )

    def forward(self, x):
        # Branch 1: 1x1 conv
        y1 = self.activation(self.b1(x))
        
        # Branch 2: 1x1 conv -> 3x3 conv
        y2 = self.activation(self.b2(x))
        y2 = self.activation(self.b2_3x3(y2))
        
        # Branch 3: 1x1 conv -> 5x5 conv (implemented as two 3x3 convs)
        y3 = self.activation(self.b3(x))
        y3 = self.activation(self.b3_5x5_1(y3))
        y3 = self.activation(self.b3_5x5_2(y3))
        
        # Branch 4: 3x3 pool -> 1x1 conv
        y4 = self.b4_pool(x)
        y4 = self.activation(self.b4_conv(y4))
        
        return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNet(nn.Module):
    """GoogLeNet model with configurable activation function"""
    
    def __init__(self, num_classes=10, activation='relu'):
        super(GoogLeNet, self).__init__()
        self.activation_name = activation
        self.activation = get_activation(activation)
        
        # Initial layers
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
        )

        # Inception blocks
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, activation)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, activation)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, activation)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, activation)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, activation)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, activation)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, activation)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, activation)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, activation)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.activation(self.pre_layers(x))
        
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        
        out = self.a5(out)
        out = self.b5(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def googlenet(num_classes=10, activation='relu'):
    """Factory function to create GoogLeNet model"""
    return GoogLeNet(num_classes=num_classes, activation=activation)
