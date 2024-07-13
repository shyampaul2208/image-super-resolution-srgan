import torch
import torch.nn as nn
import numpy as np

def convolution_block(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def upsample_block(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels*4, 3, padding=1),
        nn.PixelShuffle(2),
        nn.PReLU(in_channels)
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return out

class Generator(nn.Module):
    def __init__(self, upscale_factor=4, num_blocks=16):
        super().__init__()
        num_upblocks = int(np.log2(upscale_factor))
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.relu = nn.PReLU(64)
        self.resblocks = nn.Sequential(*([ResidualBlock(64, 64)]* num_blocks))
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.upblocks = nn.Sequential(*([upsample_block(64)]* num_upblocks))
        self.conv3 = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        identity = self.relu(out)
        out = self.resblocks(identity)
        out = self.conv2(out)
        out = self.bn(out)
        out += identity
        out = self.upblocks(out)
        out = self.conv3(out)
        return torch.tanh(out)

    
class DiscConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, idx):
        super(DiscConvLayer, self).__init__()
        self.rep_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(1 + idx % 2), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.rep_layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=64, features=[64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        blocks = []
        for i, feature in enumerate(features):
            blocks.append(DiscConvLayer(in_channels, feature, idx=i+1))
            in_channels = feature
        self.disc_blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()   
         
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.disc_blocks(x)
        return self.classifier(x)


