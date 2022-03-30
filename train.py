#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.backends.cudnn as cudnn
import argparse
import logging
import pdb

from dataprep import *
from utils import *
from models import *



# In[7]:

def train(traindataload,net,optimizer, grad_sc,epoch,device):
        
        loss = 0
        epoch_loss = 0
        size=0
        acc=0
        nbatch = len(traindataload)
        for idx, (inputs, targets) in enumerate(traindataload):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                size += outputs.shape[0]*outputs.shape[2]*outputs.shape[3]
                batch_loss = dice_loss(outputs.softmax(1),targets, multiclass=True)
                optimizer.zero_grad()
                grad_sc.scale(batch_loss).backward()
                grad_sc.step(optimizer)
                grad_sc.update()
                loss = batch_loss.item()
                epoch_loss += loss
                acc += (outputs.argmax(1) == targets.argmax(1)).type(torch.float).sum().item()
                progress_bar(idx, len(traindataload), 'Loss: %.5f, Dice-Coef: %.5f'
                         %(loss, 1-loss))#(loss/(idx+1)), (1-(loss/(idx+1)))))
                log_msg = '\n'.join(['Training (%d/%d): Epoch: %d, Loss: %.5f,  Dice-Coef:  %.5f' %(idx,nbatch, epoch,loss, 1-loss)])
                logging.info(log_msg)
        epoch_loss /= nbatch
        acc /= size
        print(f"Training epoch error: \n Acc: {(100*acc):>0.1f}%, Avg loss: {epoch_loss:>8f}, Dice: {(1-epoch_loss):>8f} \n")
        
def validation(validationload,net,epoch,device):
        loss = 0
        acc = 0
        epoch_loss=0
        size=0
        nbatch = len(validationload)
        net.eval()
        with torch.no_grad():
                for idx, (inputs, targets) in enumerate(validationload):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = net(inputs)
                        batch_loss = dice_loss(outputs.softmax(1),targets, multiclass=True).item()
                        loss = batch_loss
                        epoch_loss +=loss
                        acc += (outputs.argmax(1) == targets.argmax(1)).type(torch.float).sum().item()
                        size += outputs.shape[0]*outputs.shape[2]*outputs.shape[3]
            
                        
                        progress_bar(idx, len(validationload), 'Loss: %.5f, Dice-Coef: %.5f' %(loss, 1-loss))
                        log_msg = '\n'.join(['Validation (%d/%d): Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'
                %(idx, nbatch, epoch, loss, 1-loss)])
                        logging.info(log_msg)
        
        epoch_loss /= nbatch
        acc /= size
        net.train()                
        print(f"Validation error: \n Acc: {(100*acc):>0.1f}%, Avg loss: {loss:>8f}, Dice: {(1-loss):>8f} \n")
        return epoch_loss
                                
def trainmodel(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    cudnn.benchmark = True
    get_logger()
    
    traindataload = data_load(args, datatype='Train')
    validationload = data_load(args, datatype = 'Validation')
    net,optimizer, scheduler, best_score, start_epoch =load_model(args, mode='train')   #load_model(args, class_num=4, mode='train')
    net=net.to(device)
    grad_sc = torch.cuda.amp.GradScaler(enabled=True)
    dice =1
    nodice =0
    for epoch in range(start_epoch, start_epoch+args.epochs):
        # Train Model
        print(f'Epoch: {epoch}')
        train(traindataload,net,optimizer, grad_sc,epoch,device)
        print('\n\n<Validation>')
        validloss = validation(validationload,net,epoch,device)
       
        # Save Model
        try:
                os.makedirs(args.ckpt_root)
        except:
                pass
        #loss /= (idx+1)
        score = 1 - validloss



        if dice > validloss:
                    
                dice = validloss
                nodice=0
                best_wei = net.state_dict()
                    
        elif dice < validloss:
                nodice+=1
                if nodice >3:
                        early_stop=True
                        checkpoint = Checkpoint(net, optimizer, epoch, score)
                        checkpoint.save(os.path.join(args.ckpt_root, args.model+'.pth'))

                        #torch.save(best_wei, os.path.join(args.ckpt_root, args.model+'early_stop'+'.pth'))
                        return validloss
        #if score > best_score:
        
        checkpoint = Checkpoint(net, optimizer, epoch, score)
        checkpoint.save(os.path.join(args.ckpt_root, args.model+'.pth'))
        #best_score = score
        print("Saving...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Trianing resume.")
    parser.add_argument("--model", type=str, default='unet',
                        help="Model Name (unet,unetpp,pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--in_channel", type=int, default=4,
                        help="A number of images to use for input")
    parser.add_argument("--batch_size", type=int, default=13,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=40,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0,
                        help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.00740,
                        help="Learning rate to use in training")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--file_path", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/train/",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--file_path_validation", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/validation/",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--file_path_test", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/test/",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--output_root", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/predictionUnetPlusPlus/",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_root", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/Unet_Final_NoDP/",
                        help="The directory containing the checkpoint files")
    args = parser.parse_args()
    print(args)
    trainmodel(args)


# In[5]:





# In[ ]:




