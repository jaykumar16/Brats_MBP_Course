#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# https://github.com/cv-lee/BraTs/tree/master/pytorch: The Unet model was changed to fit our desired models 

import pdb
import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.nn.functional import softmax


def conv3x3(in_c, out_c, kernel_size=3, stride=1, padding=1,
            bias=True, useBN=True, drop_rate=0):
    if useBN:
        return nn.Sequential(
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True),
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(),
               # nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU())



class down(nn.Module):

        def __init__(self,in_c, out_c, drop_rate=0,kernel_size=3, stride=1, padding=1,
            bias=False ):
                super().__init__()
                self.d = nn.Sequential(nn.MaxPool2d(2),
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True),
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True))


        def forward(self,x):

                return self.d(x)

class Up(nn.Module):

        def __init__(self,in_c, out_c,bias=True, drop_rate=0, kernel_size=3, stride=1, padding=1,
            ):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_c,out_c, kernel_size=2, stride=2 )
                self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True),
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(out_c, out_c, kernel_size, stride, padding=1, bias=bias),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True))

                
                
        
        def forward(self,x1,x2):
                x1 = self.up(x1)
                # input is CHW
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        
class UNet(nn.Module):
    def __init__(self, in_channel=4, class_num=4,layer_multipler=32, n_layers=4, useBN=False, drop_rate=0):
        
        super(UNet, self).__init__()
        
        
        nb_filter=[]
        for i in range(0, n_layers):
                nb_filter.append(layer_multipler*(2**i))
                
        self.output_dim = class_num
        self.drop_rate = drop_rate
        self.n_layers = n_layers
        
        print(nb_filter)
        if n_layers == 5:
                self.conv1 = conv3x3(in_channel, nb_filter[0], useBN=useBN, drop_rate=self.drop_rate)
                self.conv2 = down(nb_filter[0], nb_filter[1], drop_rate=self.drop_rate)
                self.conv3 = down(nb_filter[1], nb_filter[2], drop_rate=self.drop_rate)
                self.conv4 = down(nb_filter[2], nb_filter[3], drop_rate=self.drop_rate)
                self.conv5 = down(nb_filter[3], nb_filter[4], drop_rate=self.drop_rate)

                
                        
                self.upsample54 = Up(nb_filter[4], nb_filter[3], drop_rate=self.drop_rate)
                self.upsample43 = Up(nb_filter[3], nb_filter[2], drop_rate=self.drop_rate)
                self.upsample32 = Up(nb_filter[2], nb_filter[1], drop_rate=self.drop_rate)
                self.upsample21 = Up(nb_filter[1], nb_filter[0], drop_rate=self.drop_rate)
                self.conv0  = nn.Sequential(#nn.ReflectionPad2d(1),
                                    nn.Conv2d(nb_filter[0], self.output_dim, kernel_size=1),
                                    )
                
        elif n_layers==4:
                
                self.conv1 = conv3x3(in_channel, nb_filter[0], useBN=useBN, drop_rate=self.drop_rate)
                self.conv2 = down(nb_filter[0], nb_filter[1], drop_rate=self.drop_rate)
                self.conv3 = down(nb_filter[1], nb_filter[2],  drop_rate=self.drop_rate)
                self.conv4 = down(nb_filter[2], nb_filter[3],  drop_rate=self.drop_rate)
        
                
                        
                self.upsample43 = Up(nb_filter[3], nb_filter[2], drop_rate=self.drop_rate)
                self.upsample32 = Up(nb_filter[2], nb_filter[1], drop_rate=self.drop_rate)
                self.upsample21 = Up(nb_filter[1], nb_filter[0], drop_rate=self.drop_rate)
                self.conv0  = nn.Sequential(#nn.ReflectionPad2d(1),
                                    nn.Conv2d(nb_filter[0], self.output_dim, kernel_size=1),
                                    )
                
        elif n_layers ==3:
                
                self.conv1 = conv3x3(in_channel, nb_filter[0], useBN=useBN, drop_rate=self.drop_rate)
                self.conv2 = down(nb_filter[0], nb_filter[1], drop_rate=self.drop_rate)
                self.conv3 = down(nb_filter[1], nb_filter[2], drop_rate=self.drop_rate)
                
                
                        
                self.upsample32 = Up(nb_filter[2], nb_filter[1], drop_rate=self.drop_rate)
                self.upsample21 = Up(nb_filter[1], nb_filter[0], drop_rate=self.drop_rate)        
                self.conv0  = nn.Sequential(#nn.ReflectionPad2d(1),
                                    nn.Conv2d(nb_filter[0], self.output_dim, kernel_size=1),
                                    )
        
                ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.normal_(m.weight.data, mean=0, std=0.01)

    def forward(self, x):
    
        if self.n_layers ==5.:
                output1 = self.conv1(x)
                output2 = self.conv2(output1)
                output3 = self.conv3(output2)
                output4 = self.conv4(output3)
                output5 = self.conv5(output4)

                convUp = self.upsample54(output5, output4)
                convUp = self.upsample43(convUp, output3)
                convUp = self.upsample32(convUp, output2)
                convUp = self.upsample21(convUp, output1)
                
                final = self.conv0(convUp)
                

                
        elif self.n_layers==4.:
                
                output1 = self.conv1(x)
                output2 = self.conv2(output1)
                output3 = self.conv3(output2)
                output4 = self.conv4(output3)
                
                convUp = self.upsample43(output4, output3)
                convUp = self.upsample32(convUp, output2)
                convUp = self.upsample21(convUp, output1)
                
                final = self.conv0(convUp)
                
                
        elif self.n_layers==3.:
                
                output1 = self.conv1(x)
                output2 = self.conv2(output1)
                output3 = self.conv3(output2)
                convUp = self.upsample32(output3, output2)
                convUp = self.upsample21(convUp, output1)
                
                final = self.conv0(convUp)
                

                
        return final


def test():
    net = UNet(class_num=3)
    y = net(torch.randn(3,1,240,240))
    print(y.size())


