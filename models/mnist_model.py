import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.zsize = 74
        self.network = nn.Sequential(
            # FC. 1024 RELU. batchnorm
            nn.ConvTranspose2d(self.zsize, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # FC. 7x7x128 RELU. batchnorm
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True), 
            # 4x4 upconv, 64 RELU, stride 2. batchnorm. 1 padding for 28x28 restoration.
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 4x4 upconv. 1 channel.
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        img = self.network(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.nchannels = 1
        self.convs = nn.Sequential(
            # 4x4 conv. 64 lRELU. stride 2
            nn.Conv2d(self.nchannels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # 4x4 conv. 128 lRELU. stride 2. batchnorm
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # FC. 1024 lRELU. batchnorm.
            nn.Conv2d(128, 1024, 7, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        # Input x : 28x28 Gray image.
        x = self.convs(x)
        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        # output channel would be 1
        # FC output layer for D

        self.layers = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # output channel would be 1
        output = self.layers(x)
        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()
        # FC.128 - batchnorm - 1RELU - FC.output for Q
        # output channel would be 10, 2, 2 for disc, mu, var
        self.main = nn.Sequential(
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fcmu = nn.Conv2d(128, 2, 1)
        self.fcvar = nn.Conv2d(128, 2, 1)
        self.fclogits = nn.Conv2d(128, 10, 1)
    
    def forward(self, x):
        output = self.main(x)
        disc_logits, mu, var = self.fclogits(output).squeeze(), self.fcmu(output).squeeze(), self.fcvar(output).squeeze()
        # output channel would be 10, 2, 2 for disc, mu, var
        return disc_logits, mu, torch.exp(var)