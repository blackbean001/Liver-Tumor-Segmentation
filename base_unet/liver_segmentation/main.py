import os 
import time
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
from data_util import *
from model import *
from datetime import datetime
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn

"""
https://www.kaggle.com/c/carvana-image-masking-challenge
"""

parser = argparse.ArgumentParser(description='HandAge')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 8)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--start-epoch', type=int, default=0, 
                    help='start epoch')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=212,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str, default=None,
                    help='resume training')
args = parser.parse_args()
args.cuda =torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def train(epoch, model, optimizer, train_loader, iters):
    model.train()
#    criterion=nn.NLLLoss2d(torch.FloatTensor(CLASS_WEIGHT)).cuda()
    #criterion=nn.NLLLoss2d().cuda()

    dice_co=0
    count=0
    tar = 0   
    for batch_idx,(data,target) in enumerate(train_loader):
        data = Variable(data.cuda())
        target = Variable(target.cuda())  # torch.cuda.IntTensor
        output = model(data)       # torch.cuda.FloatTensor
        output = output.float()   # torch.cuda.floatTensor

        optimizer.zero_grad()
        loss = SoftDiceLoss().forward(output[:,1,:,:], target)  
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(output,dim=1) # torch.cuda.FloatTensor
        dice_coef = compute_dice(pred, target)
        dice_co += dice_coef     

        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
                
        if batch_idx % args.log_interval == 0 and not batch_idx==0 :
 
            print('Train Epoch:{}/{} [{}/{} ({:.0f}%)]  Loss:{:.4f} acc:{:.2f}% ave dice coef:{:.4f}'.format(
                epoch, args.epochs, batch_idx *
                len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader), loss.data[0], 100.0 *
                (count+.0) / args.log_interval / tar,
                dice_co/args.log_interval
            ))

            iters += 1
            dice_co = 0
            count=0
           
    return loss.data[0],iters

def compute_dice(pred,target):
    """
    compute dice coefficient
    """
    dice_count = torch.sum(pred.data.type(torch.LongTensor) * target.data.type(torch.LongTensor))
    dice_sum = (1.0 * torch.sum(target.data.type(torch.LongTensor))) + (1.0 * torch.sum(pred.data.type(torch.LongTensor)))
    return (2 * dice_count+0.1)/(0.1 + dice_sum)

def save_checkpoint(state, is_best,filename= ROOT + 'unet_code/dice_liver_crop/checkpoints_bigsmall/checkpoint_none.pth.tar'):
    """
    save checkpoint
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, ROOT+'unet_code/dice_liver_crop/checkpoints_bigsmall/model_best_none.pth.tar')

def resume(ckpt,model):
    """
    resume training 
    """
    if os.path.isfile(ckpt):
        print('==> loading checkpoint {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']
        iters=checkpoint['iters']
        print("==> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        return model,optimizer,args.start_epoch,best_loss,iters
    else:
        print("==> no checkpoint found at '{}'".format(args.resume))
    
def adjust_lr(optimizer,epoch,decay=3):
    """
        adjust the learning rate initial lr decayed 10 every 20 epoch
    """
    lr=args.lr*(0.3**(epoch//decay))
    if epoch > 12:
        lr = args.lr*(0.3**(epoch//decay))
    for param in optimizer.param_groups:
        param['lr']=lr

def main():
    kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}
    HandSet = LITS(ROOT, TRAIN, MASK)
    
    train_loader = DataLoader(HandSet,
                              shuffle=True,
                              batch_size=args.batch_size,
                              **kwargs)

    model = uNet(NUM_CLASS)
    model.apply(weights_init) 
    
    if args.cuda:
        model.cuda()
    
    optimizer=optim.RMSprop(model.parameters(), lr=args.lr)

    best_loss=1e+5
    iters=0
    # resume training 
    if args.resume:
        model,optimizer,args.start_epoch,best_loss,iters = resume(args.resume,model)

    for epoch in range(args.start_epoch ,args.epochs):
        adjust_lr(optimizer,epoch)
        t1=time.time()
        loss, iters = train(epoch,
                            model,
                            optimizer,
                            train_loader,
                            iters)
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        state={
            'epoch':epoch,
            'state_dict':model.state_dict(),
            'optimizer':optimizer,
            'loss':best_loss,
            'iters': iters,
        }
        save_checkpoint(state, is_best)
    writer.close()

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform(m.weight.data)
        except:
            1
        
if __name__ == '__main__':
    main()

