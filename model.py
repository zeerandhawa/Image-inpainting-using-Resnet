import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

#bias = FALSE

# Define block
class resBlock(nn.Module):
    def __init__(self, inchannel, outchannel, downsample=False, upsample=False):


        super(resBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
        stride1 = 1
        self.reflect_padding = 1
        self.reflect_padding2 = 1
        self.outch1 = inchannel

        if (self.downsample == True):
            stride1 = 2
            self.reflect_padding = 1
            self.outch1 = outchannel
            self.conv1 = nn.Conv2d(inchannel, self.outch1, kernel_size=3, stride=stride1, padding=0, bias = False)
            self.conc1 = nn.Conv2d(inchannel, self.outch1, kernel_size=1, stride=2, padding=0, bias = False)

        if (self.upsample == True):
            stride1 = 2
            self.reflect_padding = 1
            self.outch1 = outchannel
            self.conv1 = nn.ConvTranspose2d(inchannel, self.outch1, kernel_size=3, stride=stride1, padding=3, output_padding =1, bias = False)
            self.conc1 = nn.Sequential(nn.Conv2d(inchannel, self.outch1, kernel_size=1, stride=1, padding=0, bias = False),
                                       nn.Upsample(scale_factor=2, mode='nearest'))

        if (self.upsample == False and self.downsample == False):
            self.conv1 = nn.Conv2d(inchannel, self.outch1, kernel_size=3, stride=stride1, padding=0)

        self.pad1 = nn.ReplicationPad2d(self.reflect_padding)
        #self.conv1 = nn.Conv2d(inchannel, self.outch1, kernel_size=3, stride = stride1, padding =0)
        self.batch1 = nn.BatchNorm2d(self.outch1)
        self.relu1 = nn.ReLU()

        self.pad2 = nn.ReplicationPad2d(self.reflect_padding2)
        self.conv2 = nn.Conv2d(self.outch1, self.outch1, kernel_size=3, stride = 1, padding =0, bias = False)
        self.batch2 = nn.BatchNorm2d(self.outch1)

        self.relu2 = nn.ReLU()



    def forward(self, x):

        # TODO: forward
        input = x

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.batch2(x)

        if(self.downsample == False and self.upsample == False):
            conc = input
        else:
            conc = self.conc1(input)

        out = x+conc
        out = self.relu2(out)
        return out


# Define network
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.pad1 = nn.ReplicationPad2d(3)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride = 1, padding =0, bias = False)
        self.relu1 = nn.ReLU()
        self.pad2 = nn.ReplicationPad2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0, bias = False)
        self.relu2 = nn.ReLU()

        self.block1 = resBlock(64, 128, downsample=True)

        self.block2 = resBlock(128, 128)

        self.block3 = resBlock(128, 128)

        self.block4 = resBlock(128, 128)
        self.block5 = resBlock(128, 128)
        self.block6 = resBlock(128, 128)
        self.block7 = resBlock(128, 128)

        self.block8 = resBlock(128, 64, upsample=True)
        self.pad3 = nn.ReplicationPad2d(1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding = 1, padding=4, bias = False)
        self.relu3 = nn.ReLU()
        self.pad4 = nn.ReplicationPad2d(3)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0, bias = True)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        # TODO: forward
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.block8(x)

        x = self.pad3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pad4(x)
        x = self.conv4(x)
        x = self.sig(x)
        return x





