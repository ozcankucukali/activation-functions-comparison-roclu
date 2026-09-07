import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation

# Model Configurations
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """VGG model with configurable activation function"""
    
    def __init__(self, vgg_name='VGG16', num_classes=10, activation='relu'):
        super(VGG, self).__init__()
        self.activation_name = activation
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    get_activation(self.activation_name)
                ]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11(num_classes=10, activation='relu'):
    """VGG-11 model"""
    return VGG('VGG11', num_classes=num_classes, activation=activation)

def VGG13(num_classes=10, activation='relu'):
    """VGG-13 model"""
    return VGG('VGG13', num_classes=num_classes, activation=activation)

def VGG16(num_classes=10, activation='relu'):
    """VGG-16 model"""
    return VGG('VGG16', num_classes=num_classes, activation=activation)

def VGG19(num_classes=10, activation='relu'):
    """VGG-19 model"""
    return VGG('VGG19', num_classes=num_classes, activation=activation)
