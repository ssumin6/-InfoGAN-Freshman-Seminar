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
        self.linear = nn.Sequential(
            # FC. 1024 RELU. batchnorm
            nn.Linear(self.zsize, 1024, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            # FC. 7x7x128 RELU. batchnorm
            nn.Linear(1024, 7*7*128, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(7*7*128)
        )
        self.transconv1 = nn.ConvTranspose2d(128, 64, 4, 2, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)
        self.transconv2 = nn.ConvTranspose2d(64, 1, 4, bias=False)

    def forward(self, x):
        x = x.view(-1, self.zsize)
        img = self.linear(x)
        # resize the tensor into shape [batch_size, 7, 7, 128]
        img = img.view(-1, 128, 7, 7)
        # 4x4 upconv, 64 RELU, stride 2. batchnorm.
        img = self.batchnorm(F.relu(self.transconv1(img)))
        # 4x4 upconv. 1 channel.
        img = torch.sigmoid(self.transconv2(img))
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.nchannels = 1
        self.convs = nn.Sequential(
            # 4x4 conv. 64 lRELU. stride 2
            nn.Conv2d(self.nchannels, 64, 4, stride=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
            # 4x4 conv. 128 lRELU. stride 2. batchnorm
            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128)
        )
        self.batchnorm = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        # Input x : 28x28 Gray image.
        x = self.convs(x)
        # flatten x
        shape = list(x.size())
        flatten_shape = shape[1]*shape[2]*shape[3]
        x = x.view(-1, flatten_shape)
        # FC. 1024 lRELU. batchnorm.
        fc = nn.Linear(flatten_shape, 1024, bias=False)
        x = self.batchnorm(F.leaky_relu(fc(x), 0.1))
        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        # output channel would be 1
        # FC output layer for D
        self.fc = nn.Linear(1024, 1, bias=False)

    def forward(self, x):
        # output channel would be 1
        output = torch.sigmoid(self.fc(x))
        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()
        # FC.128 - batchnorm - 1RELU - FC.output for Q
        # output channel would be 10, 2, 2 for disc, mu, var
        self.main = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fcmu = nn.Linear(128, 2, bias=False)
        self.fcvar = nn.Linear(128, 2, bias=False)
        self.fclogits = nn.Linear(128, 10, bias=False)
        


    def forward(self, x):
        output = self.main(x)
        disc_logits, mu, var = self.fclogits(output), self.fcmu(output), self.fcvar(output)
        # output channel would be 10, 2, 2 for disc, mu, var

        return disc_logits, mu, var