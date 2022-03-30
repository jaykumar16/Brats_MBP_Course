#!/usr/bin/env python

# This model was adopted using: https://github.com/4uiiurz1/pytorch-nested-unet
# The model is changed to adapt for different layers and size of the model

import pdb
import torch
import torch.nn as nn
import numpy as np

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, drop_rate):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out=self.dropout(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes=4, input_channels=4,layer_multipler=32,n_layers=4, drop_rate=0,deep_supervision=False, **kwargs):
        super().__init__()
        self.n_layers = n_layers

        nb_filter=[]
        for i in range(0, n_layers):
                nb_filter.append(layer_multipler*(2**i))
	
      
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if self.n_layers == 4.:
            self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],drop_rate=drop_rate)
            self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],drop_rate=drop_rate)
            self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],drop_rate=drop_rate)
      
            self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], drop_rate=drop_rate)
            self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], drop_rate=drop_rate)
      
            self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], drop_rate=drop_rate)
      
            self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

      

        elif self.n_layers==5.:
            self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],drop_rate=drop_rate)
            self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],drop_rate=drop_rate)
            self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],drop_rate=drop_rate)
            self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],drop_rate=drop_rate)

            self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], drop_rate=drop_rate)
            self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], drop_rate=drop_rate)
            self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],drop_rate=drop_rate)

            self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], drop_rate=drop_rate)
            self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2],drop_rate=drop_rate)

            self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1],drop_rate=drop_rate)

            self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0],drop_rate=drop_rate)
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        elif self.n_layers==3.:

            self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],drop_rate=drop_rate)
            self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],drop_rate=drop_rate)
      
            self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], drop_rate=drop_rate)
      
            self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], drop_rate=drop_rate)
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

      
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
      

    def forward(self, input):
        if self.n_layers==4:
            x0_0 = self.conv0_0(input)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
	    
            output = self.final(x0_3)
            

        elif self.n_layers==5:
            
            x0_0 = self.conv0_0(input)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

            
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
            output = self.final(x0_4)
            

        elif self.n_layers==3:
            
            x0_0 = self.conv0_0(input)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

           
            output = self.final(x0_2)
       
        return output

       
