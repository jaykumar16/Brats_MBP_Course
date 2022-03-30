import os
import sys

import torch
from torch.optim import Adam, SGD, RMSprop

from models import *
from models.unet import *
from .UnetPlusPlus import NestedUNet

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *


def load_model(args, mode):

    # Device Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layer_multipler = 22
    n_layers = 4
    # Model Init
    if args.model == 'unet':
        net = UNet(4,4, layer_multipler= layer_multipler, n_layers=n_layers, useBN=True,
                   drop_rate=args.drop_rate)
    elif args.model == 'unetpp':
    	net= NestedUNet(num_classes= 4, input_channels=4,layer_multipler= layer_multipler, n_layers=n_layers,
                   drop_rate=args.drop_rate)   
    else:
        raise ValueError('args.model ERROR')

    # Optimizer Init
    if mode == 'train':
        resume = args.resume
        optimizer = Adam(net.parameters(), lr=args.lr)
        #optimizer = RMSprop(net.parameters(), lr=args.lr)
        #optimizer = SGD(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
        
    elif mode == 'test':
        resume = True
        optimizer = None
    else:
        raise ValueError('load_model mode ERROR')

    # Model Load
    if resume:
        scheduler = None
        
        checkpoint = Checkpoint(net, optimizer)
        checkpoint.load(os.path.join(args.ckpt_root, args.model+'.tar'))
        best_score = checkpoint.best_score
        start_epoch = checkpoint.epoch+1
    else:
        best_score = 0
        start_epoch = 1

    if device == 'cuda:0':
        net.cuda()
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark=True

    return net,optimizer, scheduler, best_score, start_epoch
