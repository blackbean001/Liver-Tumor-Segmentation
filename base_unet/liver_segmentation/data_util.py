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

ROOT='/data/home/jasonsli/'
TRAIN='pngdata/volume'
MASK = 'pngdata/mask/liver_mask'

TEST='data/volume/test'
TEST_MASK='data/mask/lesion_mask'
NUM_CLASS=2
CLASS_WEIGHT=[1,3]

class LITS(DATA.Dataset):
    def __init__(self,root,train,mask=None):
        self.trainable=trainable
        self.train=os.path.join(root,train)
        if self.trainable:
            self.mask=os.path.join(root,mask)

        none_img = []
        for path in os.listdir(self.mask):
            img = cv2.imread(os.path.join(self.mask, path))
            if np.sum(img) != 0:
                none_img.append(path)

        self.image_list = none_img
#        self.image_list = [i for i in self.image_list if int(i.split('-')[0])<105]
        self.transform = transform

    def __getitem__(self,index):
        img_path = os.path.join(self.train, self.image_list[index])
        mask_name = self.image_list[index]
        mask_path = os.path.join(self.mask, mask_name)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        scale = random.randint(450,600)
        img = img.resize((scale,scale))
        img = np.array(img.convert('RGB'))
            #print('img',type(img))
#            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#            img = clahe.apply(img[:,:,0])

        img=img[:,:,0]/255.0

        mask = mask.resize((scale,scale))
        mask = np.array(mask)
        if len(mask.shape) == 2:
            mask = mask
        if len(mask.shape) == 3:
            mask = mask[:,:,0]

        mask = (mask > 0) + 0
            
        w,h = mask.shape
        if w>512:
            crop_out_size = 512
            a, b = random.randint(0,w-crop_out_size),random.randint(0,w-crop_out_size)
            img = img[a:(a+crop_out_size),b:(b+crop_out_size)]
            mask = mask[a:(a+crop_out_size),b:(b+crop_out_size)]
        if w<512:
            out_img = np.zeros((512,512))
            out_mask = np.zeros((512,512))
            img = img.astype(np.uint8)
            mask = mask.astype(np.uint8)

            img = cv2.resize(img,(int(256+w/2)-int(256-w/2),int(256+h/2)-int(256-h/2)))
            mask = cv2.resize(mask,(int(256+w/2)-int(256-w/2),int(256+h/2)-int(256-h/2)))

            out_img[int(256-w/2):int(256+w/2),int(256-h/2):int(256+h/2)] = img
            out_mask[int(256-w/2):int(256+w/2),int(256-h/2):int(256+h/2)] = mask
                
            img = out_img
            mask = out_mask            

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
            
        return img[np.newaxis,:,:].copy(), mask.copy()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        logits = logits.contiguous()

        intersection = (logits * targets)
        a = 2 * (intersection.sum() + smooth)
        b = ((logits.sum() + targets.sum() + smooth))
        a, b = a.float(), b.float()
        score = (a+smooth) / (b+smooth)
        score = 1 -  score
        
        return score
