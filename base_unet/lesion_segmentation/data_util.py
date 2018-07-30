import os
import torch
import cv2
import math
import torchvision.datasets
import numpy as np
#import preprocessing as pre
import torch.utils.data as DATA
from PIL import Image
from torch.utils.data import DataLoader
from skimage.segmentation import find_boundaries
import torch.nn as nn
import torch.nn.functional as F
import random


ROOT='/cephfs/group/youtu/gaia/jasonsli/liver/liver_2dunet/'
#ROOT = '/data/liver/liver_2dunet/'
TRAIN='data2/train_crop_liver_ROI_pad_withne/train/img'
MASK = 'data2/train_crop_liver_ROI_pad_withne/train/mask'

TEST='data/volume/test'
#TEST_MASK='data/mask/liver_mask'
TEST_MASK='data/mask/lesion_mask'
NUM_CLASS=2
CLASS_WEIGHT=[1,10]
#CLASS_WEIGHT= None

class LITS(DATA.Dataset):
    def __init__(self,root,train,mask=None,trainable=True):
        self.trainable=trainable
        self.train=os.path.join(root,train)
        if self.trainable:
            self.mask=os.path.join(root,mask)
        
        self.image_list = os.listdir(self.train)

    def __getitem__(self,index):
        img_path = os.path.join(self.train, self.image_list[index])
        mask_name = self.image_list[index]
        mask_path = os.path.join(self.mask, mask_name)
            
        img = Image.open(img_path)
        angle = random.randint(-45,45)
        mask = Image.open(mask_path)
        img = img.rotate(angle)
        mask = mask.rotate(angle)
            
        img = img.resize((240,240))           
        mask = mask.resize((240,240))
        img = np.array(img.convert('RGB'))

        img = cv2.fastNlMeansDenoisingColored(img, None,5,10,5,21)

        img = img.astype(np.float32)
            #print('img',type(img))
        img/=255.0
        mask = np.array(mask)
        if len(mask.shape) == 2:
            mask = mask
        if len(mask.shape) == 3: 
            mask = mask[:,:,0]

        mask = (mask > 0) + 0
        mask = mask.astype(np.float32)
        w,h = mask.shape
        #print('mask', type(mask))
        if np.random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        crop_out_size = 224
        a, b = random.randint(0,w-crop_out_size),random.randint(0,w-crop_out_size)
        img = img[a:(a+crop_out_size),b:(b+crop_out_size),:]
        mask = mask[a:(a+crop_out_size),b:(b+crop_out_size)]

        return np.transpose(img,[2,0,1]).copy(), mask.copy()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
#        num = targets.size(0)
        logits = logits.contiguous()
#        probs = F.sigmoid(logits)
#        m1 = probs.view(num, -1)
#        m1 = logits.view(num, -1)
#        m2 = targets.view(num, -1)
        intersection = (logits * targets)
        a = 2 * (intersection.sum() + smooth)
        b = ((logits.sum() + targets.sum() + smooth))
        a, b = a.float(), b.float()
        score = (a+smooth) / (b+smooth)
        score = 1 -  score
        
        return score
