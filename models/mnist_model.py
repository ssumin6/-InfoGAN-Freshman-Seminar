import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO : implement generator
        # eg.) transposed conv

    def forward(self, x):
        # TODO : implement generator
        img = None

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO : implement discriminator

    def forward(self, x):
        # TODO : implement discriminator

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO : implement discriminator head
        # output channel would be 1

    def forward(self, x):
        # TODO : implement discriminator head
        # output channel would be 1
        output = None
        
        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO : implement Q head
        # output channel would be 10, 2, 2 for disc, mu, var


    def forward(self, x):
        disc_logits, mu, var = None, None, None
        # TODO : implement Q head
        # output channel would be 10, 2, 2 for disc, mu, var

        return disc_logits, mu, var