import torch 
import torch.nn as nn

from transformers import PreTrainedModel

import math

class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.3))

    def forward(self, x):
        return super().forward(x)
    
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size=1024):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()
       
        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1)), 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)# - 1 * (kernel[1] // 2 - 1)
            padding.append((temp_pad, 0))

        # feature_height = input_height // stride[0]

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])
    
    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(0.3)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out
    
class ChannelNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporalBlock = TemporalBlock(
            in_channels=1,
            out_channels=10,
            n_layers=4,
            kernel_size=(1, 17),
            stride=(1, 2),
            dilation_list=[(1, 1), (1, 2), (1, 4), (1, 8)],
        )
        
        self.spatialBlock = SpatialBlock(
            in_channels=40,
            out_channels=40, 
            num_spatial_layers=3,
            stride=(2, 1),
            #input_height=128,
            input_height=18,
        )

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(
                        in_channels=120,
                        out_channels=120
                    ),
                    ConvLayer2D(
                        in_channels=120, 
                        out_channels=120, 
                        kernel=3, 
                        stride=2, 
                        padding=1, 
                        dilation=1
                    ),
                )
                for i in range(1)
            ]
        )

        self.final_conv = ConvLayer2D(
            in_channels=120,
            out_channels=120,
            kernel=3,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.final_pool= nn.AdaptiveAvgPool2d((1, 1))   # Global average pooling
        self.fc = nn.Linear(120, 1)
    
    def forward(self, x):
        out = self.temporalBlock(x)

        out = self.spatialBlock(out)

        for res_block in self.res_blocks:
            out = res_block(out)

        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
            
        return out
''' 
x = torch.randn(1, 1, 18, 1024) +
model = ChannelNet()
out = model(x)
print(out.shape)  
'''