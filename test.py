#!/usr/bin/env python
import argparse
import torch
import torch.backends.cudnn as cudnn
import cfl
from dataprepaug import *
from utils import *
from models import *
from sklearn.metrics import confusion_matrix

def test(args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        cudnn.benchmark = True
        testdataload = data_load(args, datatype='Test')
        
        
        net, _, _, _, _ = load_model(args, class_num=4, mode='test')
        net = net.to(device)
        net.eval()
        torch.set_grad_enabled(False)
        predication = []
        GT = []
        GTImages = []
        loss=0
        acc=0
        size=0
        nbatch = len(testdataload)
        for idx, (inputs, targets, paths) in enumerate(testdataload):
        
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                #outputs = outputs.softmax(1)
                GTImages.append(inputs.cpu().detach().numpy())
                predication.append(outputs.softmax(1).cpu().detach().numpy())
                GT.append(targets.cpu().detach().numpy())
                loss += dice_loss(outputs.softmax(1),targets, multiclass=True).item()
                acc += (outputs.argmax(1) == targets.argmax(1)).type(torch.float).sum().item()
                size += outputs.shape[0]*outputs.shape[2]*outputs.shape[3]
                
                log_msg = '\n'.join(['Test (%d/%d): Loss: %.5f,  Dice-Coef:  %.5f' %(idx, nbatch, loss/(idx+1), 1-(loss/(idx+1)))])
                logging.info(log_msg)
        
        loss /= nbatch
        acc /= size
        
        print(np.shape(GT))
        print(f"Test error: \n Acc: {(100*acc):>0.1f}%, Avg loss: {loss:>8f}, Dice: {(1-loss):>8f} \n")
                ## Save Predicted Mask
        #confusion = confusion_matrix(GT, predication)
        #print('Confusion Matrix\n')
        #print(confusion)        
        
        # Save files
        try:
                os.makedirs(args.output_root)
        except:
                pass
        cfl.writecfl(args.output_root+'GTMask', np.array(GT))
        cfl.writecfl(args.output_root+'predication', np.array(predication))
        cfl.writecfl(args.output_root+'GTImages', np.array(GTImages))
        
                
                
                
                
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--model", type=str, default='unetpp', # Need to be fixed
                        help="Model Name")
        parser.add_argument("--batch_size", type=int, default=5, # Need to be fixed
                        help="The batch size to load the data")
        parser.add_argument("--in_channel", type=int, default=4,
                        help="A number of images to use for input")
        parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
        parser.add_argument("--file_path_test", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/test/",
                        help="The directory containing the Testing image dataset.")
        parser.add_argument("--img_root", type=str, default="",
                        help="The directory containing the training image dataset.")
        parser.add_argument("--output_root", type=str, default="/home/jay/Documents/courses/Aicourse/Brats/Prediction/",
                        help="The directory containing the results.")
        parser.add_argument("--ckpt_root", type=str, default="/home/jay/Desktop/home/jay/Documents/courses/Aicourse/Brats/checkpointUnetPlusPluslr_0001/",help="The directory containing the checkpoint files")
        args = parser.parse_args()
        
        test(args)
        
                
                
        
