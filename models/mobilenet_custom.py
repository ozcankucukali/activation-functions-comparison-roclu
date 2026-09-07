import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block"""
    def __init__(self, in_planes, out_planes, stride=1, activation='relu'):
        super(DepthwiseSeparableConv, self).__init__()
        self.activation = get_activation(activation)
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, 
                                  padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, 
                                  padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.activation(self.bn1(self.depthwise(x)))
        out = self.activation(self.bn2(self.pointwise(out)))
        return out

class MobileNet(nn.Module):
    """MobileNet model with configurable activation function"""
    
    # Configuration: (out_planes, stride)
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10, activation='relu', width_multiplier=1.0):
        super(MobileNet, self).__init__()
        self.activation_name = activation
        self.activation = get_activation(activation)
        
        # Apply width multiplier
        input_channel = int(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        self.layers = self._make_layers(input_channel, width_multiplier)
        
        # Final feature size after width multiplier
        final_channel = int(1024 * width_multiplier)
        self.linear = nn.Linear(final_channel, num_classes)

    def _make_layers(self, in_planes, width_multiplier):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            
            # Apply width multiplier
            out_planes = int(out_planes * width_multiplier)
            
            layers.append(DepthwiseSeparableConv(in_planes, out_planes, stride, self.activation_name))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobileNetV2Block(nn.Module):
    """MobileNetV2 Inverted Residual Block"""
    def __init__(self, in_planes, out_planes, stride, expand_ratio, activation='relu'):
        super(MobileNetV2Block, self).__init__()
        self.stride = stride
        self.activation = get_activation(activation)
        
        hidden_dim = in_planes * expand_ratio
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_planes, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ])
        
        layers.extend([
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
        ])
        
        # Pointwise linear
        layers.extend([
            nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    """MobileNetV2 model with configurable activation function"""
    
    def __init__(self, num_classes=10, activation='relu', width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        self.activation_name = activation
        self.activation = get_activation(activation)
        
        input_channel = int(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # Building inverted residual blocks
        # [expand_ratio, out_planes, num_blocks, stride]
        self.cfg = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        self.layers = self._make_layers(input_channel, width_multiplier)
        
        # Final convolution
        final_channel = int(1280 * width_multiplier)
        self.conv2 = nn.Conv2d(int(320 * width_multiplier), final_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(final_channel)
        
        self.linear = nn.Linear(final_channel, num_classes)

    def _make_layers(self, in_planes, width_multiplier):
        layers = []
        for expand_ratio, out_planes, num_blocks, stride in self.cfg:
            out_planes = int(out_planes * width_multiplier)
            for i in range(num_blocks):
                stride = stride if i == 0 else 1
                layers.append(MobileNetV2Block(in_planes, out_planes, stride, expand_ratio, self.activation_name))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.activation(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def MobileNetV1(num_classes=10, activation='relu', width_multiplier=1.0):
    """MobileNetV1 model"""
    return MobileNet(num_classes=num_classes, activation=activation, width_multiplier=width_multiplier)

def MobileNetV2_model(num_classes=10, activation='relu', width_multiplier=1.0):
    """MobileNetV2 model"""
    return MobileNetV2(num_classes=num_classes, activation=activation, width_multiplier=width_multiplier)

def mobilenet_cifar(num_classes=10, activation='relu'):
    """MobileNet optimized for CIFAR"""
    return MobileNet(num_classes=num_classes, activation=activation, width_multiplier=0.5)
