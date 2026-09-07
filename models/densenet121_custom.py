import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation

class Bottleneck(nn.Module):
    """DenseNet Bottleneck block"""
    def __init__(self, in_planes, growth_rate, activation='relu'):
        super(Bottleneck, self).__init__()
        self.activation = get_activation(activation)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    """Transition layer between dense blocks"""
    def __init__(self, in_planes, out_planes, activation='relu'):
        super(Transition, self).__init__()
        self.activation = get_activation(activation)
        
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(self.activation(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    """DenseNet model with configurable activation function"""
    
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, activation='relu'):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.activation_name = activation
        self.activation = get_activation(activation)

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        # Dense Block 1
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans1 = Transition(num_planes, out_planes, activation)
        num_planes = out_planes

        # Dense Block 2
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans2 = Transition(num_planes, out_planes, activation)
        num_planes = out_planes

        # Dense Block 3
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans3 = Transition(num_planes, out_planes, activation)
        num_planes = out_planes

        # Dense Block 4
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate, self.activation_name))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(self.activation(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(num_classes=10, activation='relu'):
    """DenseNet-121 model"""
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, 
                   num_classes=num_classes, activation=activation)

def DenseNet169(num_classes=10, activation='relu'):
    """DenseNet-169 model"""
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32,
                   num_classes=num_classes, activation=activation)

def DenseNet201(num_classes=10, activation='relu'):
    """DenseNet-201 model"""
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32,
                   num_classes=num_classes, activation=activation)

def DenseNet161(num_classes=10, activation='relu'):
    """DenseNet-161 model"""
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48,
                   num_classes=num_classes, activation=activation)

def densenet_cifar(num_classes=10, activation='relu'):
    """DenseNet optimized for CIFAR (smaller version)"""
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12,
                   num_classes=num_classes, activation=activation)
