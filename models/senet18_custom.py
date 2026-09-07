import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16, activation='relu'):
        super(SEBlock, self).__init__()
        self.activation = get_activation(activation)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
    
    def forward(self, x):
        # Global Average Pooling
        w = F.avg_pool2d(x, x.size(2))
        # Squeeze
        w = self.activation(self.fc1(w))
        # Excitation
        w = torch.sigmoid(self.fc2(w))
        # Scale
        return x * w

class PreActBlock(nn.Module):
    """Pre-activation Block with SE module"""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.activation = get_activation(activation)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        
        # SE Block
        self.se = SEBlock(planes, activation=activation)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.activation(self.bn1(x))
        shortcut = self.shortcut(out)
        
        out = self.conv1(out)
        out = self.conv2(self.activation(self.bn2(out)))
        
        # Apply SE block
        out = self.se(out)
        
        out += shortcut
        return out

class BasicSEBlock(nn.Module):
    """Basic SE Block (post-activation)"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(BasicSEBlock, self).__init__()
        self.activation = get_activation(activation)
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE Block
        self.se = SEBlock(planes, activation=activation)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE block
        out = self.se(out)
        
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class SENet(nn.Module):
    """SENet Model with configurable activation function"""
    
    def __init__(self, block, num_blocks, num_classes=10, activation='relu'):
        super(SENet, self).__init__()
        self.in_planes = 64
        self.activation_name = activation
        self.activation = get_activation(activation)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation_name))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def SENet18(num_classes=10, activation='relu', use_preact=False):
    """SENet-18 model"""
    if use_preact:
        return SENet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, activation=activation)
    else:
        return SENet(BasicSEBlock, [2, 2, 2, 2], num_classes=num_classes, activation=activation)

def SENet34(num_classes=10, activation='relu', use_preact=False):
    """SENet-34 model"""
    if use_preact:
        return SENet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes, activation=activation)
    else:
        return SENet(BasicSEBlock, [3, 4, 6, 3], num_classes=num_classes, activation=activation)
