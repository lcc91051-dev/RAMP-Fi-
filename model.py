import random
import numpy as np
import torch
from torch import nn

def conv3x3(in_channel, out_channel, stride=1, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an optional residual projection."""
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identify = self.downsample(x)
        
        out += identify
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers, norm_layer=None,data_inchannel=3):
        super(ResNet, self).__init__()
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_channel = 64
        self.conv1 = nn.Conv2d(data_inchannel, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def _make_layer(self, block, out_channel, blocks, stride=1):
        """Create one residual layer of the encoder."""
        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.in_channel != out_channel*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel*block.expansion, stride),
                norm_layer(out_channel*block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample, norm_layer))

        self.in_channel = out_channel*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channel, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class RAMPFi(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.encoder = ResNet(BasicBlock, [2, 2, 2, 2], data_inchannel=2)
        self.classifier = nn.Linear(512, 6, bias=bias)

    def forward(self, x):
        # Encode each synchronous ratio view with shared parameters.
        h = torch.stack((self.encoder(x[:, 0]), self.encoder(x[:, 1])), dim=1)
        return h, self.classifier(h)
class CSIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=ResNet(BasicBlock,[2,2,2,2],data_inchannel=2)
        self.classifier=nn.Linear(512,6)
    def forward(self,x):return self.classifier(self.encoder(x))
